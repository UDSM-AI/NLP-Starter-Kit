# app.py

import streamlit as st
from inference import classify_news

def main():
    st.title("Swahili News Classifier App")

    # Add a text area for user input
    user_input = st.text_area("Enter the Swahili news article text:")

    # Add a button to trigger the classification
    if st.button("Classify"):
        if user_input:
            # Call your news classifier function
            result_dict, highest_prob = classify_news(user_input)

            # Display probabilities for all categories
            st.write("Prediction Probabilities:")
            for category, probability in result_dict.items():

                # Display category and probability
                st.write(f"{category}: {probability}")
                # Convert probability to a percentage
                percentage = float(probability[:-1])  # Remove the '%' and convert to float
                # Display progress bar
                st.progress(percentage / 100.0)


            # Display the highest predicted category
            # st.success(f"Predicted category: {highest_prob}")

        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
