
    window.stack_plot_probe_1 = {
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
                "label": "Stack plot for the first proximity probe",
                "data": [
                    1.2572850962346445,
                    1.257134634529185,
                    1.2540712926199982,
                    1.2574411506582228,
                    1.2572637264133406
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
    