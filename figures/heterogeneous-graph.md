# Heterogeneous Graph - Horizontal Flowchart

This diagram represents a heterogeneous graph with activity and object nodes in separate grouped boxes.

## Features
- Horizontal layout with activity nodes on the left, object nodes on the right
- Larger round nodes for both activities and objects
- Undirected action-object (gaze) edges
- Loops between Grocery Bag ↔ Fridge Drawer and Tomato ↔ Knife
- Inverted Cutting Board → Knife direction
- Consistent edge styling with different colors for edge types
- Grouped nodes in dashed boxes for visual organization
- Clean edges without labels for better readability

```mermaid
flowchart LR
    %% Node styling with larger size
    classDef activity fill:#FF8C00,stroke:#FF6347,stroke-width:3px,color:#000,font-size:16px
    classDef object fill:#90EE90,stroke:#228B22,stroke-width:3px,color:#000,font-size:16px

    %% Activity nodes
    A0(("Open<br/>Fridge")):::activity
    A1(("Cut<br/>Tomato")):::activity

    %% Object nodes
    O0(("Fridge<br/>Drawer")):::object
    O1(("Grocery<br/>Bag")):::object
    O2(("Tomato")):::object
    O3(("Knife")):::object
    O4(("Cutting<br/>Board")):::object

    %% Edges without labels
    %% Temporal edges (orange)
    A0 --> A1
    linkStyle 0 stroke:#FF6347,stroke-width:3px

    %% Gaze edges (black, undirected)
    A0 --- O0
    linkStyle 1 stroke:#000000,stroke-width:3px
    A0 --- O1
    linkStyle 2 stroke:#000000,stroke-width:3px
    A1 --- O2
    linkStyle 3 stroke:#000000,stroke-width:3px
    A1 --- O3
    linkStyle 4 stroke:#000000,stroke-width:3px
    A1 --- O4
    linkStyle 5 stroke:#000000,stroke-width:3px

    %% Relation edges (green)
    O0 --> O1
    linkStyle 6 stroke:#228B22,stroke-width:3px
    O1 --> O0
    linkStyle 7 stroke:#228B22,stroke-width:3px
    O1 --> O2
    linkStyle 8 stroke:#228B22,stroke-width:3px
    O2 --> O3
    linkStyle 9 stroke:#228B22,stroke-width:3px
    O3 --> O2
    linkStyle 10 stroke:#228B22,stroke-width:3px
    O4 --> O3
    linkStyle 11 stroke:#228B22,stroke-width:3px
    O2 --> O4
    linkStyle 12 stroke:#228B22,stroke-width:3px
```

## Edge Types Legend

- **Temporal edges** (orange): Connect activities in temporal sequence
- **Gaze edges** (black, undirected): Connect activities to objects that receive gaze attention
- **Relation edges** (green): Connect objects that have spatial or functional relationships

## Node Types

- **Activity nodes** (orange, round): Represent human activities with time duration in brackets, grouped in dashed box
- **Object nodes** (green, round): Represent physical objects in the environment, grouped in dashed box
