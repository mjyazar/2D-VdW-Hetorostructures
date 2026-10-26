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
mose2 = MoSe2()
wse2 = WSe2()

graphene_mose2 = GrapheneMDBilayer(graphene.monolayer(), mose2.monolayer())
graphene_wse2 = GrapheneMDBilayer(graphene.monolayer(), wse2.monolayer())


SYSTEMS = []

SYSTEMS.append(SystemSettings("Graphene_Bilayer", "Graphene", "Graphene", graphene, (15, 15, 1), (75, 75, 1), (6, 30, 3), (24, 100, 6)))
SYSTEMS.append(SystemSettings("Graphene_MoSe2", "Graphene", "MoSe2", graphene_mose2, (6, 6, 1), (21, 21, 1), (6, 12, 3), (12, 30, 6)))
SYSTEMS.append( SystemSettings("Graphene_WSe2", "Graphene", "WSe2", graphene_wse2, (6, 6, 1), (21, 21, 1), (6, 12, 3), (12, 30, 6)))
