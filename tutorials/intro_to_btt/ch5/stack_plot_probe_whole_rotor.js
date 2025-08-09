
    window.stack_plot_probe_whole_rotor = {
    "type": "line",
    "data": {
        "labels": [
            "AoA_1",
            "AoA_2",
            "AoA_3",
            "AoA_4",
            "AoA_5"
        ],
        "datasets": [
            {
                "label": "Stack plot for probe 1",
                "data": [
                    1.2572850962346445,
                    1.257134634529185,
                    1.2540712926199982,
                    1.2574411506582228,
                    1.2572637264133406
                ]
            },
            {
                "label": "Stack plot for probe 2",
                "data": [
                    1.2572189924631163,
                    1.2572884286144934,
                    1.2573327024206105,
                    1.253932417877842,
                    1.2573885838033942
                ]
            },
            {
                "label": "Stack plot for probe 3",
                "data": [
                    1.2571918747080462,
                    1.2573333016499293,
                    1.2572386361914627,
                    1.2538911674381161,
                    1.2574870447885025
                ]
            },
            {
                "label": "Stack plot for probe 4",
                "data": [
                    1.2571021456703573,
                    1.257251585396776,
                    1.2572948173990732,
                    1.2539831337843417,
                    1.257551409304428
                ]
            }
        ]
    },
    "options": {
        "plugins": {
            "annotation": {},
            "zoom": {
                "pan": {
                    "enabled": true,
                    "modifierKey": "ctrl"
                },
                "zoom": {
                    "drag": {
                        "enabled": true
                    },
                    "mode": "xy"
                }
            }
        },
        "scales": {
            "x": {
                "title": {
                    "display": true,
                    "text": "Blade no",
                    "font": {
                        "size": 20
                    }
                },
                "ticks": {}
            },
            "y": {
                "title": {
                    "display": true,
                    "text": "AoA difference (rad)",
                    "font": {
                        "size": 20
                    }
                },
                "ticks": {}
            }
        }
    }
};
    