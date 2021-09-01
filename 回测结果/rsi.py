# type: ignore

from datetime import datetime

from backtrader.utils import AutoOrderedDict, OrderedDict

# 每年的收益率
OrderedDict(
    [
        (datetime.date(2015, 12, 31), 0.10000499999999946),
        (datetime.date(2016, 12, 31), 0.060516997650010396),
        (datetime.date(2017, 12, 31), 0.009637622645454869),
        (datetime.date(2018, 12, 31), 0.019021630694752822),
        (datetime.date(2019, 12, 31), 0.04766538829098943),
        (datetime.date(2020, 12, 31), 0.060717495208480354),
        (datetime.date(2021, 12, 31), -0.0005420692199155086),
    ]
)
# 收益率统计
OrderedDict(
    [
        ("rtot", 0.2874733006604657),
        ("ravg", 1.0314423618114374e-05),
        ("rnorm", 0.044811305671620484),
        ("rnorm100", 4.481130567162048),
    ]
)

# 夏普比率
OrderedDict([("sharperatio", 1.2985848796463144)])

# 交易细节
AutoOrderedDict(
    [
        ("total", AutoOrderedDict([("total", 1002), ("open", 1), ("closed", 1001)])),
        (
            "streak",
            AutoOrderedDict(
                [
                    ("won", AutoOrderedDict([("current", 0), ("longest", 8)])),
                    ("lost", AutoOrderedDict([("current", 1), ("longest", 11)])),
                ]
            ),
        ),
        (
            "pnl",
            AutoOrderedDict(
                [
                    (
                        "gross",
                        AutoOrderedDict(
                            [
                                ("total", 1663.0650000000112),
                                ("average", 1.6614035964036076),
                            ]
                        ),
                    ),
                    (
                        "net",
                        AutoOrderedDict(
                            [
                                ("total", 1663.0650000000112),
                                ("average", 1.6614035964036076),
                            ]
                        ),
                    ),
                ]
            ),
        ),
        (
            "won",
            AutoOrderedDict(
                [
                    ("total", 444),
                    (
                        "pnl",
                        AutoOrderedDict(
                            [
                                ("total", 5687.4050000000025),
                                ("average", 12.809470720720727),
                                ("max", 108.2099999999997),
                            ]
                        ),
                    ),
                ]
            ),
        ),
        (
            "lost",
            AutoOrderedDict(
                [
                    ("total", 557),
                    (
                        "pnl",
                        AutoOrderedDict(
                            [
                                ("total", -4024.339999999994),
                                ("average", -7.225026929982035),
                                ("max", -96.77999999999969),
                            ]
                        ),
                    ),
                ]
            ),
        ),
        (
            "long",
            AutoOrderedDict(
                [
                    ("total", 501),
                    (
                        "pnl",
                        AutoOrderedDict(
                            [
                                ("total", 1168.5150000000117),
                                ("average", 2.3323652694611012),
                                (
                                    "won",
                                    AutoOrderedDict(
                                        [
                                            ("total", 3363.3800000000056),
                                            ("average", 14.687248908296967),
                                            ("max", 108.2099999999997),
                                        ]
                                    ),
                                ),
                                (
                                    "lost",
                                    AutoOrderedDict(
                                        [
                                            ("total", -2194.8649999999925),
                                            ("average", -8.069356617647031),
                                            ("max", -96.77999999999969),
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                    ("won", 229),
                    ("lost", 272),
                ]
            ),
        ),
        (
            "short",
            AutoOrderedDict(
                [
                    ("total", 500),
                    (
                        "pnl",
                        AutoOrderedDict(
                            [
                                ("total", 494.549999999998),
                                ("average", 0.9890999999999961),
                                (
                                    "won",
                                    AutoOrderedDict(
                                        [
                                            ("total", 2324.024999999999),
                                            ("average", 10.809418604651158),
                                            ("max", 85.49999999999969),
                                        ]
                                    ),
                                ),
                                (
                                    "lost",
                                    AutoOrderedDict(
                                        [
                                            ("total", -1829.475),
                                            ("average", -6.419210526315789),
                                            ("max", -29.890000000000327),
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                    ("won", 215),
                    ("lost", 285),
                ]
            ),
        ),
        (
            "len",
            AutoOrderedDict(
                [
                    ("total", 27486),
                    ("average", 27.458541458541458),
                    ("max", 478),
                    ("min", 1),
                    (
                        "won",
                        AutoOrderedDict(
                            [
                                ("total", 16945),
                                ("average", 38.164414414414416),
                                ("max", 478),
                                ("min", 1),
                            ]
                        ),
                    ),
                    (
                        "lost",
                        AutoOrderedDict(
                            [
                                ("total", 10541),
                                ("average", 18.9245960502693),
                                ("max", 345),
                                ("min", 1),
                            ]
                        ),
                    ),
                    (
                        "long",
                        AutoOrderedDict(
                            [
                                ("total", 18857),
                                ("average", 37.63872255489022),
                                ("max", 478),
                                ("min", 1),
                                (
                                    "won",
                                    AutoOrderedDict(
                                        [
                                            ("total", 11798),
                                            ("average", 51.519650655021834),
                                            ("max", 478),
                                            ("min", 1),
                                        ]
                                    ),
                                ),
                                (
                                    "lost",
                                    AutoOrderedDict(
                                        [
                                            ("total", 7059),
                                            ("average", 25.952205882352942),
                                            ("max", 345),
                                            ("min", 1),
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                    (
                        "short",
                        AutoOrderedDict(
                            [
                                ("total", 8629),
                                ("average", 17.258),
                                ("max", 216),
                                ("min", 1),
                                (
                                    "won",
                                    AutoOrderedDict(
                                        [
                                            ("total", 5147),
                                            ("average", 23.93953488372093),
                                            ("max", 216),
                                            ("min", 1),
                                        ]
                                    ),
                                ),
                                (
                                    "lost",
                                    AutoOrderedDict(
                                        [
                                            ("total", 3482),
                                            ("average", 12.217543859649123),
                                            ("max", 69),
                                            ("min", 1),
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                ]
            ),
        ),
    ]
)
