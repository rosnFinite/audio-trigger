"""
Dash component for a calibration accordion panel. Component is to be used as children of a
dash_mantine_components.AccordionItem
"""
from dash_mantine_components import (AccordionPanel,
                                     Affix,
                                     Badge,
                                     Text,
                                     Space,
                                     Center,
                                     TextInput,
                                     NumberInput,
                                     Paper,
                                     Stack,
                                     ButtonGroup,
                                     Button, Code)
from dash_iconify import DashIconify
from dash.dcc import Graph

CalibrationPanel = AccordionPanel(
    children=[
        Text([
            "In gleichen Abstand zu einer Audioquelle das "
            "Schallpegelmessgerät und Mikrofon aufstellen. Messungen zu "
            "unterschiedlichen dB(A)-Pegeln durchführen und Werte des Mikrofons "
            "abspeichern."
        ]),
        Space(h=10),
        Text([
            "Es werden mindesten 3 Messungen empfohlen. Bei Abschluss der ",
            "Kalibrierung wird diese in ein JSON-Format unter ",
            Code("/calibration"),
            "gespeichert."
        ]),
        Space(h=10),
        Center(
            Paper(
                p="md",
                withBorder=True,
                shadow="md",
                children=[
                    Center(
                        [
                            NumberInput(
                                id="db-value",
                                label="dB(A)-Wert",
                                description="Gemessen über Schallpegelmessgerät",
                                value=40,
                                min=0,
                                step=5,
                                style={"width": 270}
                            ),
                            Space(w=20),
                            Stack(
                                [
                                    Text("Mikrofonwert", size="md"),
                                    Text(id="microphone-value"),
                                ]
                            )
                        ]
                    ),
                    Space(h=10),
                    Center(
                        ButtonGroup(
                            children=[
                                Button(
                                    "Speichern",
                                    disabled=True,
                                    id="save-value-button",
                                    color="green"
                                ),
                                Button(
                                    "Zurücksetzen",
                                    id="reset-value-button",
                                    color="yellow"
                                )
                            ]
                        )
                    ),
                ]
            ),
        ),
        Space(h=20),
        Graph(id='db-microphone-graph'),
        Space(h=20),
        Stack(
            [
                Center(
                    TextInput(
                        id="calib-filename",
                        label="Name der Kalibrierungsdatei",
                        style={"width": 400}
                    )
                ),
                Button(
                    "Kalibrierung abspeichern",
                    disabled=True,
                    id="save-calibration-button",
                    color="green"
                )
            ]
        ),
    ],
)


DataPanel = AccordionPanel(
    [
        Center(
            ButtonGroup(
                children=[
                    Button("Start",
                               id="start-button",
                               size="lg",
                               color="green",
                               disabled=True,
                               leftIcon=DashIconify(icon="solar:play-outline"), ),
                    Button("Stop",
                               id="stop-button",
                               size="lg",
                               color="red",
                               disabled=True,
                               leftIcon=DashIconify(icon="solar:stop-outline"), ),
                    Button("Trigger",
                               id="trigger-button",
                               size="lg",
                               color="yellow",
                               disabled=True,
                               leftIcon=DashIconify(
                                   icon="solar:align-horizonta-spacing-bold"), )
                ]
            )
        ),
        Space(h=20),
        Graph(
            id='signal-graph'
        ),
        Space(h=20),
        Graph(
            id='frequency-graph'
        ),
        Space(h=20),
        Graph(
          id="heatmap"
        ),
        Affix(
            Badge(id="note-text",
                      size="xl",
                      variant="gradient",
                      gradient={"from": "teal", "to": "lime", "deg": 105}, ),
            position={"bottom": 20, "right": 20}
        )
    ]
)
