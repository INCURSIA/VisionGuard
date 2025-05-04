import json
import os
import ollama

folder_path = 'frame_jsons'  

batch_summaries = []
file_list = [f for f in os.listdir(folder_path) if f.endswith('.json')]

for i in range(0, len(file_list), 3):
    batch_files = file_list[i:i+3]
    combined_object_summary = ""

    for filename in batch_files:
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r') as file:
            json_data = json.load(file)

        frame_number = json_data["frame"]
        objects = json_data["objects"]

        object_summary = "\n".join([f"Frame {frame_number}: Object ID {obj['id']} is a {obj['label']} at position {obj['position']}. Description: {obj['description']}" for obj in objects])

        combined_object_summary += object_summary + "\n\n"

    # Refined Prompt with context for LLaVA
    prompt = f"""
You are a police officer watching surveillance footage, and your task is to interpret the data coming from different frames.

Each frame represents a 2-second segment of footage, and you are given a set of objects detected within each frame. Each object is assigned an ID, has a label (such as 'car', 'person', etc.), a position (a set of coordinates representing its bounding box), and a description. The description provides additional context, but it is for reference only. Your goal is to analyze the changes in the positions of these objects over time and interpret the behavior based on their movements.

Consider the following details for each frame:
1. The **frame number** represents a specific moment in time.
2. Each **object** has an ID, label, position, and a description.
3. The **position** refers to the bounding box coordinates of the object, which should help you track how objects move across the frames. Pay particular attention to objects that change position drastically or quickly within a short time.

For each batch of frames, please summarize the changes in the scene. Focus on the following:
- Any **significant movements** or drastic changes in position of objects between frames.
- **Object interactions**, if any (e.g., cars coming close to pedestrians, objects moving closer or farther from each other).
- **Notable behavior patterns** based on the object labels and positions.
- Use the descriptions for reference, but focus on how the objects' positions and movements align with real-world behavior.

Here is the data from frames {batch_files}:

{combined_object_summary}

Now, please summarize the overall behavior and movements of the objects across the frames provided, focusing on significant changes and interactions. Make sure to describe how the objects’ positions evolve and any notable patterns that emerge across the frames.
"""

    # Generate the summary from LLaVA
    response = ollama.chat(model="llava", messages=[{"role": "user", "content": prompt}])

    batch_summary = response['message']['content']
    batch_summaries.append(batch_summary)

    # Save the batch summary to a JSON file
    batch_filename = f'batch_summary_{i//3 + 1}.json'  # Save with a unique name for each batch
    batch_file_path = os.path.join("report", batch_filename)
    
    # Create the batch summary dictionary
    batch_data = {
        "batch_files": batch_files,
        "summary": batch_summary
    }

    # Write the batch summary to a JSON file
    with open(batch_file_path, 'w') as batch_file:
        json.dump(batch_data, batch_file, indent=4)

# Combine all batch summaries into a final summary
final_combined_summary = "\n".join(batch_summaries)

final_prompt = f"""
You are a senior officer reviewing the reports from your subordinates who are monitoring surveillance footage of an incident. The data they have provided pertains to different time frames, with each report focusing on object movement and interactions between different frames. Your duty is to review these reports and provide a coherent, professional summary based on the observed events.

Here is the combined summary of objects across all frames:

{final_combined_summary}

Your task is to:
1. Summarize the overall behavior and movements of the objects across all frames, focusing on significant changes and interactions.
2. Consider any unusual or conflicting information from the reports and adjust them to make sense of the situation.
3. Identify any potential patterns, such as object interactions (e.g., cars interacting with pedestrians) or drastic changes in position that could indicate unusual events or traffic conditions.
4. If there are any discrepancies or odd behavior (such as objects appearing to move too quickly or in inconsistent directions), use your experience and logic to resolve the situation and offer an interpretation that fits within realistic scenarios.

Please provide a clear and concise final summary based on the information provided above.
"""

final_response = ollama.chat(model="llava", messages=[{"role": "user", "content": final_prompt}])

# Output the final combined summary
final_summary = final_response['message']['content']
print("Final Combined Summary:", final_summary)
