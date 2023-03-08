import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    GraphFst,
    NEMO_SPACE,
    insert_space
)

from nemo_text_processing.text_normalization.en.utils import get_abs_path

zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teen.tsv")))
tens = pynini.invert(pynini.string_file(get_abs_path("data/numbers/tens.tsv")))
tens_one = pynini.invert(pynini.string_file(get_abs_path("data/numbers/tens_one.tsv")))
hundreds = pynini.invert(pynini.string_file(get_abs_path("data/numbers/hundreds.tsv")))


class CardinalFst(GraphFst):
    '''
    Finite state transducer for classifying cardinals in Italian, e.g.
        "1000" ->  cardinal { integer: "mille" }
        "2.000.000" -> cardinal { integer: "due milioni" }
    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    '''
    def __init__(self, deterministic: bool=True):
        super().__init__(name='cardinal', kind='classify', deterministic=deterministic)

        # zero
        graph_zero = zero

        # single digit
        graph_digit = digit

        # double digit
        graph_tens = teen
        graph_tens |= tens + (pynutil.delete('0') | graph_digit)
        graph_tens |= tens_one

        self.two_digit_no_zero = pynini.union(
            graph_digit, graph_tens, (pynini.cross('0', NEMO_SPACE) + graph_digit)
        ).optimize()

        # three digit
        graph_hundreds = hundreds + pynini.union(
            pynutil.delete('00'), graph_tens, (pynini.cross("0", NEMO_SPACE) + graph_digit)
        )
        graph_hundreds |= pynini.cross('100', 'cento')
        graph_hundreds |= pynini.cross('1', 'cento') + insert_space + pynini.union(
            graph_tens, pynutil.delete("0") + graph_digit
        )

        self.hundreds = graph_hundreds.optimize()

        # three digit starting with zeros
        graph_hundreds_component = pynini.union(
            graph_hundreds, pynutil.delete("0") + graph_tens
        )

        graph_hundreds_component_at_least_one_none_zero_digit = graph_hundreds_component | (pynutil.delete("00") + graph_digit)

        graph_thousands = pynini.cross('1', 'mille') + insert_space + pynini.union(
            graph_hundreds, pynutil.delete("0") + graph_tens, pynutil.delete("00") + graph_digit
        )  
        
        graph_thousands_component_at_least_one_none_zero_digit = pynini.union(
            graph_thousands,
            pynutil.delete("000") + graph_hundreds_component_at_least_one_none_zero_digit,
            graph_hundreds_component_at_least_one_none_zero_digit
            + pynutil.insert(" mila")
            + ((insert_space + graph_hundreds_component_at_least_one_none_zero_digit) | pynutil.delete("000")),
            pynini.cross("001", "mila")
            + ((insert_space + graph_hundreds_component_at_least_one_none_zero_digit) | pynutil.delete("000")),
        )




