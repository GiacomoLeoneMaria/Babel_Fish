from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.text_normalization.en.verbalizers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.it.verbalizers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.it.verbalizers.electronic import ElectronicFst

class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)
        cardinal = CardinalFst(deterministic=deterministic)
        cardinal_graph = cardinal.fst
        electronic = ElectronicFst(deterministic=deterministic)
        electronic_graph = electronic.fst
        whitelist_graph = WhiteListFst(deterministic=deterministic).fst

        graph = (
            cardinal_graph
            | electronic_graph
            | whitelist_graph
        )

        self.fst = graph

