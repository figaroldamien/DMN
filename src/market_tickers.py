"""Ticker lists by market."""

from __future__ import annotations

from typing import Dict, List


NASDAQ100_COMPONENTS: Dict[str, Dict[str, str]] = {
    "ADBE": {"sector": "Technology", "sub_sector": "Software"},
    "AMD": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "ABNB": {"sector": "Consumer Discretionary", "sub_sector": "Travel Services"},
    "ALNY": {"sector": "Healthcare", "sub_sector": "Biotechnology"},
    "GOOGL": {"sector": "Communication Services", "sub_sector": "Internet Content"},
    "GOOG": {"sector": "Communication Services", "sub_sector": "Internet Content"},
    "AMZN": {"sector": "Consumer Discretionary", "sub_sector": "Internet Retail"},
    "AEP": {"sector": "Utilities", "sub_sector": "Regulated Electric"},
    "AMGN": {"sector": "Healthcare", "sub_sector": "Biotechnology"},
    "ADI": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "AAPL": {"sector": "Technology", "sub_sector": "Consumer Electronics"},
    "AMAT": {"sector": "Technology", "sub_sector": "Semiconductor Equipment"},
    "APP": {"sector": "Technology", "sub_sector": "Software"},
    "ARM": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "ASML": {"sector": "Technology", "sub_sector": "Semiconductor Equipment"},
    "TEAM": {"sector": "Technology", "sub_sector": "Software"},
    "ADSK": {"sector": "Technology", "sub_sector": "Software"},
    "ADP": {"sector": "Industrials", "sub_sector": "Professional Services"},
    "AXON": {"sector": "Industrials", "sub_sector": "Aerospace and Defense"},
    "BKR": {"sector": "Energy", "sub_sector": "Oil and Gas Equipment"},
    "BKNG": {"sector": "Consumer Discretionary", "sub_sector": "Travel Services"},
    "AVGO": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "CDNS": {"sector": "Technology", "sub_sector": "Software"},
    "CHTR": {"sector": "Communication Services", "sub_sector": "Telecom Services"},
    "CTAS": {"sector": "Industrials", "sub_sector": "Business Services"},
    "CSCO": {"sector": "Technology", "sub_sector": "Communication Equipment"},
    "CCEP": {"sector": "Consumer Staples", "sub_sector": "Beverages"},
    "CTSH": {"sector": "Technology", "sub_sector": "IT Services"},
    "CMCSA": {"sector": "Communication Services", "sub_sector": "Entertainment"},
    "CEG": {"sector": "Utilities", "sub_sector": "Independent Power"},
    "CPRT": {"sector": "Industrials", "sub_sector": "Business Services"},
    "CSGP": {"sector": "Real Estate", "sub_sector": "Real Estate Services"},
    "COST": {"sector": "Consumer Staples", "sub_sector": "Retail"},
    "CRWD": {"sector": "Technology", "sub_sector": "Software"},
    "CSX": {"sector": "Industrials", "sub_sector": "Railroads"},
    "DDOG": {"sector": "Technology", "sub_sector": "Software"},
    "DXCM": {"sector": "Healthcare", "sub_sector": "Medical Devices"},
    "FANG": {"sector": "Energy", "sub_sector": "Oil and Gas E&P"},
    "DASH": {"sector": "Consumer Discretionary", "sub_sector": "Internet Retail"},
    "EA": {"sector": "Communication Services", "sub_sector": "Gaming"},
    "EXC": {"sector": "Utilities", "sub_sector": "Regulated Electric"},
    "FAST": {"sector": "Industrials", "sub_sector": "Industrial Distribution"},
    "FER": {"sector": "Industrials", "sub_sector": "Engineering and Construction"},
    "FTNT": {"sector": "Technology", "sub_sector": "Software"},
    "GEHC": {"sector": "Healthcare", "sub_sector": "Medical Devices"},
    "GILD": {"sector": "Healthcare", "sub_sector": "Biotechnology"},
    "HON": {"sector": "Industrials", "sub_sector": "Conglomerates"},
    "IDXX": {"sector": "Healthcare", "sub_sector": "Diagnostics and Research"},
    "INSM": {"sector": "Healthcare", "sub_sector": "Biotechnology"},
    "INTC": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "INTU": {"sector": "Technology", "sub_sector": "Software"},
    "ISRG": {"sector": "Healthcare", "sub_sector": "Medical Devices"},
    "KDP": {"sector": "Consumer Staples", "sub_sector": "Beverages"},
    "KLAC": {"sector": "Technology", "sub_sector": "Semiconductor Equipment"},
    "KHC": {"sector": "Consumer Staples", "sub_sector": "Packaged Foods"},
    "LRCX": {"sector": "Technology", "sub_sector": "Semiconductor Equipment"},
    "LIN": {"sector": "Materials", "sub_sector": "Specialty Chemicals"},
    "MAR": {"sector": "Consumer Discretionary", "sub_sector": "Lodging"},
    "MRVL": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "MELI": {"sector": "Consumer Discretionary", "sub_sector": "Internet Retail"},
    "META": {"sector": "Communication Services", "sub_sector": "Internet Content"},
    "MCHP": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "MU": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "MSFT": {"sector": "Technology", "sub_sector": "Software"},
    "MSTR": {"sector": "Technology", "sub_sector": "Software"},
    "MDLZ": {"sector": "Consumer Staples", "sub_sector": "Packaged Foods"},
    "MPWR": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "MNST": {"sector": "Consumer Staples", "sub_sector": "Beverages"},
    "NFLX": {"sector": "Communication Services", "sub_sector": "Entertainment"},
    "NVDA": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "NXPI": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "ORLY": {"sector": "Consumer Discretionary", "sub_sector": "Auto Parts"},
    "ODFL": {"sector": "Industrials", "sub_sector": "Trucking"},
    "PCAR": {"sector": "Industrials", "sub_sector": "Heavy Trucks"},
    "PLTR": {"sector": "Technology", "sub_sector": "Software"},
    "PANW": {"sector": "Technology", "sub_sector": "Software"},
    "PAYX": {"sector": "Industrials", "sub_sector": "Professional Services"},
    "PYPL": {"sector": "Financials", "sub_sector": "Credit Services"},
    "PDD": {"sector": "Consumer Discretionary", "sub_sector": "Internet Retail"},
    "PEP": {"sector": "Consumer Staples", "sub_sector": "Beverages"},
    "QCOM": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "REGN": {"sector": "Healthcare", "sub_sector": "Biotechnology"},
    "ROP": {"sector": "Technology", "sub_sector": "Software"},
    "ROST": {"sector": "Consumer Discretionary", "sub_sector": "Retail"},
    "STX": {"sector": "Technology", "sub_sector": "Computer Hardware"},
    "SHOP": {"sector": "Technology", "sub_sector": "Software"},
    "SBUX": {"sector": "Consumer Discretionary", "sub_sector": "Restaurants"},
    "SNPS": {"sector": "Technology", "sub_sector": "Software"},
    "TMUS": {"sector": "Communication Services", "sub_sector": "Telecom Services"},
    "TTWO": {"sector": "Communication Services", "sub_sector": "Gaming"},
    "TSLA": {"sector": "Consumer Discretionary", "sub_sector": "Auto Manufacturers"},
    "TXN": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "TRI": {"sector": "Industrials", "sub_sector": "Information Services"},
    "VRSK": {"sector": "Industrials", "sub_sector": "Consulting Services"},
    "VRTX": {"sector": "Healthcare", "sub_sector": "Biotechnology"},
    "WMT": {"sector": "Consumer Staples", "sub_sector": "Retail"},
    "WBD": {"sector": "Communication Services", "sub_sector": "Entertainment"},
    "WDC": {"sector": "Technology", "sub_sector": "Computer Hardware"},
    "WDAY": {"sector": "Technology", "sub_sector": "Software"},
    "XEL": {"sector": "Utilities", "sub_sector": "Regulated Electric"},
    "ZS": {"sector": "Technology", "sub_sector": "Software"},
}

CAC40_COMPONENTS: Dict[str, Dict[str, str]] = {
    "AC.PA": {"sector": "Consumer Discretionary", "sub_sector": "Lodging"},
    "AI.PA": {"sector": "Materials", "sub_sector": "Specialty Chemicals"},
    "AIR.PA": {"sector": "Industrials", "sub_sector": "Aerospace and Defense"},
    "ALO.PA": {"sector": "Industrials", "sub_sector": "Rail Transportation"},
    "ATO.PA": {"sector": "Technology", "sub_sector": "IT Services"},
    "BN.PA": {"sector": "Consumer Staples", "sub_sector": "Packaged Foods"},
    "BNP.PA": {"sector": "Financials", "sub_sector": "Banks"},
    "CA.PA": {"sector": "Consumer Staples", "sub_sector": "Food Retail"},
    "CAP.PA": {"sector": "Technology", "sub_sector": "IT Services"},
    "CS.PA": {"sector": "Financials", "sub_sector": "Insurance"},
    "DG.PA": {"sector": "Industrials", "sub_sector": "Engineering and Construction"},
    "DSY.PA": {"sector": "Technology", "sub_sector": "Software"},
    "ACA.PA": {"sector": "Financials", "sub_sector": "Banks"},
    "EL.PA": {"sector": "Healthcare", "sub_sector": "Medical Devices"},
    "EN.PA": {"sector": "Industrials", "sub_sector": "Engineering and Construction"},
    "ENGI.PA": {"sector": "Utilities", "sub_sector": "Multi-Utilities"},
    "ERF.PA": {"sector": "Healthcare", "sub_sector": "Diagnostics and Research"},
    "GLE.PA": {"sector": "Financials", "sub_sector": "Banks"},
    "HO.PA": {"sector": "Industrials", "sub_sector": "Aerospace and Defense"},
    "KER.PA": {"sector": "Consumer Discretionary", "sub_sector": "Luxury Goods"},
    "LR.PA": {"sector": "Industrials", "sub_sector": "Electrical Equipment"},
    "MC.PA": {"sector": "Consumer Discretionary", "sub_sector": "Luxury Goods"},
    "ML.PA": {"sector": "Consumer Discretionary", "sub_sector": "Auto Parts"},
    "MT.AS": {"sector": "Materials", "sub_sector": "Steel"},
    "OR.PA": {"sector": "Consumer Staples", "sub_sector": "Personal Products"},
    "ORA.PA": {"sector": "Communication Services", "sub_sector": "Telecom Services"},
    "PUB.PA": {"sector": "Communication Services", "sub_sector": "Advertising"},
    "RI.PA": {"sector": "Consumer Staples", "sub_sector": "Beverages"},
    "RMS.PA": {"sector": "Consumer Discretionary", "sub_sector": "Luxury Goods"},
    "RNO.PA": {"sector": "Consumer Discretionary", "sub_sector": "Auto Manufacturers"},
    "SAF.PA": {"sector": "Industrials", "sub_sector": "Aerospace and Defense"},
    "SAN.PA": {"sector": "Healthcare", "sub_sector": "Drug Manufacturers"},
    "SGO.PA": {"sector": "Industrials", "sub_sector": "Building Products"},
    "STLAM.MI": {"sector": "Consumer Discretionary", "sub_sector": "Auto Manufacturers"},
    "STMPA.PA": {"sector": "Technology", "sub_sector": "Semiconductors"},
    "SU.PA": {"sector": "Industrials", "sub_sector": "Electrical Equipment"},
    "TEP.PA": {"sector": "Industrials", "sub_sector": "Business Services"},
    "TTE.PA": {"sector": "Energy", "sub_sector": "Integrated Oil and Gas"},
    "VIE.PA": {"sector": "Utilities", "sub_sector": "Water Utilities"},
    "VIV.PA": {"sector": "Communication Services", "sub_sector": "Entertainment"},
}

NASDAQ100: List[str] = list(NASDAQ100_COMPONENTS.keys())
CAC40: List[str] = list(CAC40_COMPONENTS.keys())


def tickers_by_sector_and_subsector(
    index_components: Dict[str, Dict[str, str]],
    sector: str,
    sub_sector: str,
) -> List[str]:
    """Return tickers matching both sector and sub-sector."""
    sector_norm = sector.strip().lower()
    sub_sector_norm = sub_sector.strip().lower()
    return [
        ticker
        for ticker, meta in index_components.items()
        if meta.get("sector", "").strip().lower() == sector_norm
        and meta.get("sub_sector", "").strip().lower() == sub_sector_norm
    ]


def tickers_by_sector(
    index_components: Dict[str, Dict[str, str]],
    sector: str,
) -> List[str]:
    """Return tickers matching a sector."""
    sector_norm = sector.strip().lower()
    return [
        ticker
        for ticker, meta in index_components.items()
        if meta.get("sector", "").strip().lower() == sector_norm
    ]


INDEX: List[str] = [
    "^GSPC", "^DJI","^IXIC", "^RUT", "^NYA",
    "^N225","^HSI",
    "^FTSE",  "^GDAXI", "^FCHI", "^N100", "^BFX", 
    "^BVSP",
    "CL=F","GC=F", "SI=F",
    "EURUSD=X", "GBPUSD=X", "BTC-USD",  
    #"XLF", "XLY", "XLC", "XLI", "XLE", "XLB", "XLV", "XLU", "XBI", "XHB", "XRT", "XME",
    "^TNX",      
]

TABLE8_ASSETS_BY_CATEGORY: Dict[str, List[str]] = {
    "fx": [
        "6A=F", "6B=F", "USDSGD=X", "DX-Y.NYB", "6J=F", "NZDUSD=X",
        "CADJPY=X", "6C=F", "6M=F", "USDNOK=X", "6S=F", "USDSEK=X",
    ],
    "bond": [
        # Table-8 aliases mapped to liquid Yahoo proxies when exact contracts are unavailable.
        "ZN=F", "ZB=F", "ZT=F", "ZB=F", "ZN=F", "ZT=F", "ZF=F", "ZN=F", "ZF=F",
    ],
    "index": [
        "RTY=F", "^STOXX50E", "^HSI", "^FTSE", "^STOXX50E", "^FCHI",
        "ES=F", "IJH", "^N225", "YM=F", "NQ=F", "^STOXX50E",
    ],
    "comdty": [
        # Proxies: CRB -> DBC ETF, Milk III -> DBA ETF, Minneapolis/KC wheat -> KE=F.
        "DBC", "KC=F", "KE=F", "ZR=F", "GC=F", "ZL=F", "ZO=F", "ZS=F", "ZW=F", "CC=F", "CT=F", "^SPGSCI", "KE=F",
        "SB=F", "ZC=F", "SI=F", "ZM=F", "PL=F", "LE=F", "HE=F", "DBA", "OJ=F", "LBR=F", "PA=F", "GF=F", "HG=F", "ZR=F",
    ],
    "energy": [
        "RB=F", "BZ=F", "HO=F", "NG=F", "CL=F",
    ],
}

TABLE8_ALL: List[str] = list(
    dict.fromkeys(
        ticker
        for category in ("fx", "bond", "index", "comdty", "energy")
        for ticker in TABLE8_ASSETS_BY_CATEGORY[category]
    )
)

TEST: List[str] = ["^FCHI" ]
MARKET_TICKERS: Dict[str, List[str]] = {
    "nasdaq100": NASDAQ100,
    "cac40": CAC40,
    "index": INDEX,
    "table8_all": TABLE8_ALL,
    "test": TEST,
}
