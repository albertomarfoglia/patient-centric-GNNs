from rdflib import Graph, URIRef

from utils.ontologies import NS_ONTO


class MedsGraph:
    def __init__(self, graph: Graph):
        self.graph = graph

    def enrich_event_semantic(self, select_code: URIRef, new_property: URIRef):
        for event in self.graph.subjects(predicate=NS_ONTO.hasCode, object=select_code):
            subject = next(
                self.graph.objects(subject=event, predicate=NS_ONTO.hasSubject), None
            )
            if subject is not None:
                self.graph.add((subject, new_property, event))
                self.graph.remove((event, NS_ONTO.hasSubject, subject))

    def invert_has_subject(self):
        for s, _, o in list(self.graph.triples((None, NS_ONTO.hasSubject, None))):
            self.graph.remove((o, NS_ONTO.hasSubject, s))
            self.graph.add((s, NS_ONTO.hasSubject, o))


def gen_meds_kg(
    in_graph_path,
    out_graph_path,
    enrich_by_graphs: list[Graph] = list(),
    enrich_events: dict[URIRef, URIRef] = dict(),
):
    g = Graph().parse(in_graph_path, format="nt")

    meds = MedsGraph(g)

    meds.invert_has_subject()

    for k, v in enrich_events.items():
        meds.enrich_event_semantic(select_code=k, new_property=v)

    for s in enrich_by_graphs:
        meds.graph += s

    meds.graph.serialize(destination=out_graph_path, format="nt")
