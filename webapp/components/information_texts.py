from dash_mantine_components import Text, Space, Code

IntroductionText = [Text(
        [
            "Hierbei handelt es sich um einen Prototyp zur ",
            "Visualisierung einkommender Audiosignale, ",
            "als auch für übergebene WAV-Dateien. Hierzu muss im ",
            "Folgenden entweder ein gewünschtes Eingabegerät ausgewählt werden oder ",
            "aber eine WAV-Datei übegeben werden."
        ]
    ),
    Space(h=10),
    Text(
        [
            "Je nach OS kann es vorkommen, dass gewisse Eingabegeräte keine ",
            "passende Ausgabe liefern. In diesem Fall muss das innerhalb des OS als Standard"
            "festgelegte Gerät ausgewäht werden."
            "(Der Graph wird in ",
            "einem festgelegten Intervall von 200ms aktualisiert)"
        ]
    ),
    Space(h=10),
    Text(
        [
            "Wurde eine Audiodatei übergeben so wird das Signal dieser äquivalent zum ",
            "Audiostreaming unterhalb visualisiert. Damit eine WAV-Datei ausgewählt ",
            "werden kann, muss sie im Projektordner unter ",
            Code("/audio"),
            " abgelegt werden."
        ]),
    Space(h=10),
    Text(
        [
            "Falls bereits ein Graph erstellt wurde und die Audiodatei oder ",
            "das Eingabegerät geändert werden, so wird der Graph dadurch überschrieben."
        ],
        color="blue"
    )
]