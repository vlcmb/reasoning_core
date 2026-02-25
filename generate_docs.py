import os
import ast

tasks_dir = "reasoning_core/tasks"
docs_tasks_dir = "docs/tasks"

os.makedirs(docs_tasks_dir, exist_ok=True)

task_files = [f for f in os.listdir(tasks_dir) if f.endswith(".py") and not f.startswith("_")]
task_files.sort()

for task_file in task_files:
    module_name = task_file[:-3]
    file_path = os.path.join(tasks_dir, task_file)
    md_filename = f"{module_name}.md"
    
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    
    try:
        tree = ast.parse(code)
    except SyntaxError:
        print(f"Skipping {task_file} due to syntax error.")
        continue
        
    classes_to_doc = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # Target classes that contain "Config" in their name OR inherit directly from "Task" or "Config"
            is_target = False
            if 'Config' in node.name or 'Task' in node.name:
                is_target = True
            
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in ['Task', 'Config']:
                    is_target = True
                elif isinstance(base, ast.Attribute) and base.attr in ['Task', 'Config']:
                    is_target = True
            
            # Exclude purely internal utility classes starting with _
            if is_target and not node.name.startswith('_'):
                 classes_to_doc.append(node.name)
                 
    if not classes_to_doc:
        # Fallback to whole module if heuristic failed
        content = f"# {module_name.replace('_', ' ').title()}\n\n::: reasoning_core.tasks.{module_name}\n"
    else:
        content = f"# {module_name.replace('_', ' ').title()}\n\n"
        for cls in classes_to_doc:
            content += f"::: reasoning_core.tasks.{module_name}.{cls}\n"
    
    with open(os.path.join(docs_tasks_dir, md_filename), "w", encoding="utf-8") as f:
        f.write(content)
        print(f"Generated {md_filename} with specific classes: {classes_to_doc}")
