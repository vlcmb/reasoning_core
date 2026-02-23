import os

tasks_dir = "reasoning_core/tasks"
docs_tasks_dir = "docs/tasks"

os.makedirs(docs_tasks_dir, exist_ok=True)

task_files = [f for f in os.listdir(tasks_dir) if f.endswith(".py") and not f.startswith("_")]
task_files.sort()

for task_file in task_files:
    module_name = task_file[:-3]
    md_filename = f"{module_name}.md"
    
    content = f"""# {module_name.replace('_', ' ').title()}

::: reasoning_core.tasks.{module_name}
"""
    
    with open(os.path.join(docs_tasks_dir, md_filename), "w") as f:
        f.write(content)
        print(f"Generated {md_filename}")
