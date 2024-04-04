
    window.c08_resonance_1_EO_selection = {
    "data": {
        "labels": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16
        ],
        "datasets": [
            {
                "type": "bar",
                "label": "EO Error",
                "data": [
                    9095364.801467743,
                    8866417.683425307,
                    9028659.799116548,
                    9057911.547514245,
                    7592664.619328894,
                    5521317.375270963,
                    3304460.4948963877,
                    1999233.2165753206,
                    2689919.9479860035,
                    5177453.665612246,
                    7415097.56002377,
                    8689172.458625238,
                    9010410.49745195,
                    8919922.436750786,
                    8668275.507284714,
                    8878712.983816335
                ],
                "backgroundColor": "rgba(0, 0, 0, 0.5)",
                "borderColor": "rgba(0, 0, 0, 0.5)",
                "borderWidth": 1
            }
        ]
    },
    "options": {
        "plugins": {
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
                    "text": "EO",
                    "font": {
                        "size": 20
                    }
                }
            },
            "y": {
                "title": {
                    "display": true,
                    "text": "Sum of squared error",
                    "font": {
                        "size": 20
                    }
                },
                "position": "left"
            }
        }
    }
};
    