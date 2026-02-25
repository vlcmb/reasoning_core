import streamlit as st
import reasoning_core as rcr
# --- Main area ---
st.title("Reasoning Core Interactive Dashboard")

# --- Task category selection ---
st.subheader("Step 1: Choose a Task Category")
task_categories = rcr.list_tasks()
task_category = st.selectbox(
    "Available Task Categories",
    task_categories,
    index=0,
    key="task_category",
)

task_init = rcr.get_task(task_category)()

# --- Level slider (moved here) ---
st.subheader("Step 2: Choose a Level")
level = st.slider("Difficulty Level", min_value=0, max_value=4, value=0, key="level")


if st.button("Generate Task"):
    task_init.config.set_level(level)
    problem = task_init.generate()  # should return dict with prompt, true_response, maybe others
    
    st.session_state["task"] = task_init
    st.session_state["problem"] = problem

# Display prompt if available
if "problem" in st.session_state:
    problem = st.session_state["problem"]
    task = st.session_state["task"]
    st.subheader("Problem Prompt")
    st.write(task.prompt(problem.metadata))

    # User input for answer
    user_answer = st.text_area("Your Answer:")

    if st.button("Submit Answer"):
        # scoring function
        score = task.score_answer(user_answer, problem)
        st.write(f"**Your Score:** {score}")
        st.write(f"**True Answer:** {problem.answer}")
