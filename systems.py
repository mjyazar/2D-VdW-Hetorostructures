from dataclasses import dataclass
from structures import *

@dataclass(frozen=True)
class SystemSettings:
    name: str
    bottom: str
    top: str
    structure: object
    kgrid: tuple
    kgrid_dense: tuple
    conv_scf: tuple
    conv_nscf: tuple

graphene = Graphene()
mos2 = MoS2()
ws2 = WS2()


graphene_mos2 = GrapheneHeterostructure(graphene.monolayer(), mos2.monolayer())
graphene_ws2 = GrapheneHeterostructure(graphene.monolayer(), ws2.monolayer())


SYSTEMS = []

SYSTEMS.append(SystemSettings("Graphene_Bilayer", "Graphene", "Graphene", graphene, (15, 15, 1), (75, 75, 1), (6, 30, 3), (24, 100, 6)))
# SYSTEMS.append(SystemSettings("Graphene_MoS2", "Graphene", "MoS2", graphene_mos2, (6, 6, 1), (21, 21, 1), (6, 12, 3), (12, 30, 6)))
# SYSTEMS.append(SystemSettings("Graphene_WS2", "Graphene", "WS2", graphene_ws2, (6, 6, 1), (21, 21, 1), (6, 12, 3), (12, 30, 6)))
