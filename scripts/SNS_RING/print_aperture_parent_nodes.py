"""
In `paint_production_Holmes.py`, the aperture nodes are added as child nodes, with the 
parent nodes selected by index (e.g. `nodes[27]`. But the .lat file used in this script 
(data/SNS_RING_MAD_nux6.24_nuy6.15.lat) is in MAD format; the standard MADX output
(data/SNS_RING_nux6.18_nuy6.18.lat) has a different number of nodes with different
names.
"""
from __future__ import print_function
import os

from orbit.teapot import TEAPOT_Lattice


lattice = TEAPOT_Lattice()
lattice.readMAD(
    os.path.join(os.getcwd(), "scripts/SNS_RING/data/SNS_RING_MAD_nux6.24_nuy6.15.lat"), 
    "RINGINJ",
)
lattice.initialize()

# Aperture node names (keys) and the parent node indices (values):
node_indices = {
    'ap06200': 173,
    'ap06201': 186,

    'ap10000': 15,
    'ap10001': 16,
    'ap10002': 21,
    'ap10003': 31,
    'ap10004': 45,
    'ap10005': 58,
    'ap10006': 78,
    'ap10007': 103,
    'ap10008': 116,
    'ap10009': 130,
    'ap10010': 140,
    'ap10011': 144,
    'ap10012': 147,
    'ap10013': 150,
    'ap10014': 153,
    'ap10015': 156,
    'ap10016': 158,
    'ap10017': 160,
    'ap10018': 168,
    'ap10019': 198,
    'ap10020': 199,
    'ap10021': 202,
    'ap10022': 203,
    'ap10023': 211,
    'ap10024': 225,
    'ap10025': 238,
    'ap10026': 258,
    'ap10027': 283,
    'ap10028': 296,
    'ap10029': 310,
    'ap10030': 319,
    'ap10031': 322,
    'ap10032': 330,
    'ap10033': 344,
    'ap10034': 351,
    'ap10035': 358,
    'ap10036': 359,
    'ap10037': 360,
    'ap10038': 364,
    'ap10039': 371,
    'ap10040': 372,
    'ap10041': 380,
    'ap10042': 394,
    'ap10043': 407,
    'ap10044': 427,
    'ap10045': 452,
    'ap10046': 465,
    'ap10047': 479,
    'ap10048': 489,
    'ap10049': 491,
    'ap10050': 502,
    'ap10051': 503,
    'ap10052': 504,
    'ap10053': 506,
    'ap10054': 508,
    'ap10055': 510,
    'ap10056': 514,
    'ap10057': 521,
    'ap10058': 524,
    'ap10059': 535,
    'ap10060': 549,
    'ap10061': 562,
    'ap10062': 582,
    'ap10063': 607,
    'ap10064': 620,
    'ap10065': 634,
    'ap10066': 644,
    'ap10067': 647,
    'ap10068': 651,
    'ap10069': 661,

    'ap12000': 5,
    'ap12001': 8,
    'ap12002': 172,
    'ap12003': 174,
    'ap12004': 185,
    'ap12005': 187,
    'ap12006': 341,
    'ap12007': 361,
    'ap12008': 498,
    'ap12009': 511,
    'ap12010': 658,

    'ap12500': 70,
    'ap12501': 91,
    'ap12502': 250,
    'ap12503': 271,
    'ap12504': 419,
    'ap12505': 440,
    'ap12506': 574,
    'ap12507': 595,

    'ap13000': 175,
    'ap13001': 184,
    'ap13002': 188,
    'ap13003': 189,
    'ap13004': 192,

    'ap14000': 6,
    'ap14001': 7,
    'ap14002': 169,
    'ap14003': 170,
    'ap14004': 171,
    'ap14005': 176,
    'ap14006': 180,
    'ap14007': 183,
    'ap14008': 190,
    'ap14009': 191,
    'ap14010': 342,
    'ap14011': 343,
    'ap14012': 362,
    'ap14013': 363,
    'ap14014': 500,
    'ap14015': 501,
    'ap14016': 512,
    'ap14017': 513,
    'ap14018': 659,
    'ap14019': 660,

    'apell00': 35,
    'apell01': 49,
    'apell02': 62,
    'apell03': 74,
    'apell04': 87,
    'apell05': 99,
    'apell06': 112,
    'apell07': 126,
    'apell08': 159,
    'apell09': 215,
    'apell10': 229,
    'apell11': 242,
    'apell12': 254,
    'apell13': 267,
    'apell14': 279,
    'apell15': 292,
    'apell16': 306,
    'apell17': 384,
    'apell18': 398,
    'apell19': 411,
    'apell20': 423,
    'apell21': 436,
    'apell22': 448,
    'apell23': 461,
    'apell24': 475,
    'apell25': 539,
    'apell26': 553,
    'apell27': 566,
    'apell28': 578,
    'apell29': 591,
    'apell30': 603,
    'apell31': 616,
    'apell32': 630,
}

for node in sorted(node_indices):
    index = node_indices[node]
    print(node, nodes[index].getName())