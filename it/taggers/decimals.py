import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.it.utils import get_abs_path

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)

from nemo_text_processing.text_normalization.es.graph_utils import (
    cardinal_separator,
    decimal_separator,
    strip_cardinal_apocope,
)

quantities = pynini.string_file(get_abs_path("data/numbers/quantities.tsv"))
digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))


def get_quantity(decimal_graph: "pynini.FstLike", cardinal_graph: "pynini.FstLike") -> "pynini.FstLike":
    """
    """
    numbers = pynini.closure(NEMO_DIGIT, 1, 6) @ cardinal_graph
    numbers = pynini.cdrewrite(pynutil.delete(cardinal_separator), "", "", NEMO_SIGMA) @ numbers

    res = (
        pynutil.insert('integer_part: "')
        + numbers  # The cardinal we're passing only produces 'un' for one, so gender agreement is safe (all quantities are masculine). Limit to 10^6 power.
        + pynutil.insert('"')
        + NEMO_SPACE
        + pynutil.insert('quantity: "')
        + quantities
        + pynutil.insert('"')
    )
    res |= decimal_graph + NEMO_SPACE + pynutil.insert('quantity: "') + quantities + pynutil.insert('"')
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        -2 milioni: decimal { negative: "true" integer_part: "due" quantity: "milioni" preserve_order: true } -->
            meno due milioni
    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)
        graph_digit = digit | zero

        if not deterministic:
            graph = pynini.union(graph_digit, cardinal.hundreds, cardinal.tens)
            graph += pynini.closure(insert_space + graph)

        else:
            # General pattern is 1-3 digits: map as cardinal, default to tens followed by digits otherwise \
            graph = pynini.union(
                graph_digit + pynini.closure(insert_space + zero),
                cardinal.tens + pynini.closure(insert_space + zero),
                cardinal.hundreds + pynini.closure(insert_space + zero),
                cardinal.tens
                + pynini.closure(insert_space + cardinal.tens, 1)
                + pynini.closure(insert_space + zero, 0, 1)
                + (
                    pynini.closure(insert_space + graph_digit, 0, 1) | pynini.closure(insert_space + zero, 0)
                ),  # Read out as tens and a possible trailing digit or zeroes
                zero
                + pynini.closure(insert_space + zero)
                + pynini.closure(insert_space + graph_digit),  # For cases such as "1,010"
            )

        # Technically decimals should be space delineated groups of three, e.g. (1,333 333). This removes any possible spaces
        strip_formatting = pynini.cdrewrite(delete_space, "", "", NEMO_SIGMA)
        graph = strip_formatting @ graph

        self.graph = graph.optimize()

        graph_separator = pynutil.delete(decimal_separator)
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)

        self.graph_fractional = pynutil.insert('fractional_part: "') + self.graph + pynutil.insert('"')

        graph_integer = (
            strip_cardinal_apocope(cardinal.graph)
            if deterministic
            else pynini.union(cardinal.graph, strip_cardinal_apocope(cardinal.graph))
        )

        self.graph_integer = pynutil.insert('integer_part: "') + graph_integer + pynutil.insert('"')
        final_graph_wo_sign = self.graph_integer + graph_separator + insert_space + self.graph_fractional

        self.final_graph_wo_negative = (
            final_graph_wo_sign | get_quantity(final_graph_wo_sign, cardinal.graph).optimize()
        )
        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()