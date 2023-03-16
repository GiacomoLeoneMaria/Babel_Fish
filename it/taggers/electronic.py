import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.it.utils import load_labels, get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_ALPHA,
    GraphFst,
    insert_space
)

common_domains = [x[0] for x in load_labels(get_abs_path("data/electronic/domain.tsv"))]
symbols = [x[0] for x in load_labels(get_abs_path("data/electronic/symbols.tsv"))]

class ElectronicFst(GraphFst):
    """
    """
    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        dot = pynini.accep(".")
        accepted_common_domains = pynini.union(*common_domains)
        accepted_symbols = pynini.union(*symbols) - dot
        accepted_characters = pynini.closure(NEMO_ALPHA | NEMO_DIGIT | accepted_symbols)
        acceepted_characters_with_dot = pynini.closure(NEMO_ALPHA | NEMO_DIGIT | accepted_symbols | dot)

        # e-mail
        username = (
            pynutil.insert("username: \"")
            + acceepted_characters_with_dot
            + pynutil.insert("\"")
            + pynini.cross('@', ' ')
        )

        domain_graph = accepted_characters + dot + accepted_characters
        domain_graph = (
            pynutil.insert("domain: \"")
            + domain_graph
            + pynini.closure((accepted_symbols | dot) + pynini.closure(accepted_characters, 1), 0, 1)
            + pynutil.insert("\"")
        )

        domain_common_graph = (
            pynutil.insert("domain: \"")
            + accepted_characters
            + accepted_common_domains
            + pynini.closure((accepted_symbols | dot) + pynini.closure(accepted_characters, 1), 0, 1)
            + pynutil.insert("\"")
        )

        graph = username + insert_space + (domain_graph | domain_common_graph)

        # url
        protocol_start = pynini.accep("https://") | pynini.accep("http://")
        protocol_end = (
            pynini.accep("www.")
            if deterministic
            else pynini.accep("www.") | pynini.cross("www.", "vu vu vu.")
        )
        protocol = protocol_start | protocol_end | (protocol_start + protocol_end)
        protocol = pynutil.insert("protocol: \"") + protocol + pynutil.insert("\"")
        graph |= protocol + insert_space + (domain_graph | domain_common_graph)
        self.graph = graph

        final_graph = self.add_tokens(self.graph + pynutil.insert(" preserve_order: true"))
        self.fst = final_graph.optimize()