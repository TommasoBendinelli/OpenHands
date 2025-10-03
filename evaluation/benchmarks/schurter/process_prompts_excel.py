import pandas as pd
import numpy as np
import json
import os

# read all sheets
df = pd.read_excel("validation_prompts/ValidationPrompts_Edited.xlsx")

metadata_task = {
    "id": "",
    "question": "",
    "datasources": [],
    "reasoning_task": "",
    "level": "",
    "notes": "",
    "answer": "",
}

# MachineDeviceManuals, MachineDocumentation, MachineMaintenanceLog,     MachineProcessData, MachineProcessData/doc, MachineProcessDescriptions, MachineSettings, MachineSpecificHowTos, MachineSpecificUserManual, WorkInstructions

df = df[:39]

# Iterate through each row in the dataframe
for index, row in df.iterrows():

    metadata_task['id'] = row['ID']
    metadata_task['question'] = row['Validation Prompt']
    metadata_task['reasoning_task'] = row['Reasoning Task']
    metadata_task['level'] = row['Level']
    metadata_task['notes'] = row['Notes']
    metadata_task['answer'] = row['Answer']

    if row["MachineDeviceManuals"] == 'x':
        metadata_task['datasources'].append("MachineDeviceManuals")
    if row["MachineDocumentation"] == 'x':
        metadata_task['datasources'].append("MachineDocumentation")
    if row["MachineMaintenanceLog"] == 'x':
        metadata_task['datasources'].append("MachineMaintenanceLog")
    if row["MachineProcessData"] == 'x':
        metadata_task['datasources'].append("MachineProcessData")
    if row["MachineProcessData/doc"] == 'x':
        metadata_task['datasources'].append("MachineProcessDataDoc")
    if row["MachineProcessDescriptions"] == 'x':
        metadata_task['datasources'].append("MachineProcessDescriptions")
    if row["MachineSettings"] == 'x':
        metadata_task['datasources'].append("MachineSettings")
    if row["MachineSpecificHowTos"] == 'x':
        metadata_task['datasources'].append("MachineSpecificHowTos")
    if row["MachineSpecificUserManual"] == 'x':
        metadata_task['datasources'].append("MachineSpecificUserManual")
    if row["WorkInstructions"] == 'x':
        metadata_task['datasources'].append("WorkInstructions")

    filename = f"tasks/questions/prompt_{metadata_task['id']}_metadata.json"

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(
        filename,
        "w",
    ) as f:
        json.dump(metadata_task, f, indent=4)

    print(metadata_task)

    metadata_task = {
        "id": "",
        "question": "",
        "datasources": [],
        "reasoning_task": "",
        "level": "",
        "notes": "",
        "answer": "",
    }
