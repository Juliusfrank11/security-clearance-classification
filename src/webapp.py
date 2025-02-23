import streamlit as st
from create_lr_model import create_lr_model

st.title("Security Clearance Appeal Outcome Predictor")

guideline_letter_to_description = {
    "A": "Allegiance to the United States",
    "B": "Foreign Influence",
    "C": "Foreign Preference",
    "D": "Sexual Behavior",
    "E": "Personal Conduct",
    "F": "Financial Considerations",
    "G": "Alcohol Consumption",
    "H": "Drug Involvement",
    "I": "Psychological Conditions",
    "J": "Criminal Conduct",
    "K": "Handling Protected Information",
    "L": "Outside Activities",
    "M": "Misuse of Information Technology Systems",
}

lr, train_data, test_data = {}, {}, {}

st.write("Training models...")

lr["A"], train_data["A"], test_data["A"] = create_lr_model("A", test_size=0.001)
lr["B"], train_data["B"], test_data["B"] = create_lr_model("B", test_size=0.001)
lr["C"], train_data["C"], test_data["C"] = create_lr_model("C", test_size=0.001)
lr["D"], train_data["D"], test_data["D"] = create_lr_model("D", test_size=0.001)
lr["E"], train_data["E"], test_data["E"] = create_lr_model("E", test_size=0.001)
lr["F"], train_data["F"], test_data["F"] = create_lr_model("F", test_size=0.001)
lr["G"], train_data["G"], test_data["G"] = create_lr_model("G", test_size=0.001)
lr["H"], train_data["H"], test_data["H"] = create_lr_model("H", test_size=0.001)
lr["H"], train_data["H"], test_data["H"] = create_lr_model("I", test_size=0.001)
lr["I"], train_data["I"], test_data["I"] = create_lr_model("J", test_size=0.001)
lr["J"], train_data["J"], test_data["J"] = create_lr_model("J", test_size=0.001)
lr["K"], train_data["K"], test_data["K"] = create_lr_model("K", test_size=0.001)
lr["M"], train_data["M"], test_data["M"] = create_lr_model("M", test_size=0.001)

st.write("Models trained!")


def process_user_text_input(user_input):
    for guideline, model in lr.items():
        prob_for_guideline = model.predict_proba([user_input])[0][0]
        st.write(
            "Probability of rejected appeal for guideline {} ({}): {}%".format(
                guideline,
                guideline_letter_to_description.get(guideline),
                round(prob_for_guideline * 100, 2),
            )
        )


user_input = st.text_input("Enter some text:")
if st.button("Submit"):
    process_user_text_input(user_input)
