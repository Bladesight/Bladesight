
    window.wrapped_aoa_histogram = {
    "type": "bar",
    "data": {
        "labels": [
            4.0,
            11.0,
            18.0,
            26.0,
            33.0,
            40.0,
            48.0,
            55.0,
            62.0,
            70.0,
            77.0,
            84.0,
            92.0,
            99.0,
            107.0,
            114.0,
            121.0,
            129.0,
            136.0,
            143.0,
            151.0,
            158.0,
            165.0,
            173.0,
            180.0,
            187.0,
            195.0,
            202.0,
            209.0,
            217.0,
            224.0,
            231.0,
            239.0,
            246.0,
            253.0,
            261.0,
            268.0,
            276.0,
            283.0,
            290.0,
            298.0,
            305.0,
            312.0,
            320.0,
            327.0,
            334.0,
            342.0,
            349.0,
            356.0
        ],
        "datasets": [
            {
                "label": "Histogram of AoAs",
                "data": [
                    421,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    790,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    790,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    790,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    790,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    369
                ]
            }
        ]
    },
    "options": {
        "plugins": {
            "annotation": {
                "annotations": {
                    "blade_1_label": {
                        "type": "label",
                        "xValue": 0,
                        "yValue": 400,
                        "xAdjust": 100,
                        "yAdjust": -40,
                        "backgroundColor": "rgba(255, 255, 255, 0.25)",
                        "content": [
                            "Blade 1"
                        ],
                        "textAlign": "start",
                        "font": {
                            "size": 18
                        },
                        "callout": {
                            "display": true,
                            "side": 10
                        }
                    },
                    "blade_1_wrapped": {
                        "type": "label",
                        "xValue": 48,
                        "yValue": 380,
                        "xAdjust": -150,
                        "yAdjust": -70,
                        "backgroundColor": "rgba(255, 255, 255, 0.25)",
                        "content": [
                            "Also blade 1!"
                        ],
                        "textAlign": "start",
                        "font": {
                            "size": 18
                        },
                        "callout": {
                            "display": true,
                            "side": 10
                        }
                    }
                }
            },
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
                    "text": "AoA [degrees]",
                    "font": {
                        "size": 20
                    }
                },
                "ticks": {}
            },
            "y": {
                "title": {
                    "display": true,
                    "text": "Blade count",
                    "font": {
                        "size": 20
                    }
                },
                "ticks": {}
            }
        }
    }
};
    