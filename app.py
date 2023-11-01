import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))


# Define the main function to run your Streamlit app
def main():

    # Add a title and description for your app
    st.title("Predicting Hotel Cancellations using Machine Learning")

    no_of_adults = st.number_input('Number of adults',step=1, min_value=0)
    no_of_children = st.number_input('Number of children',step=1, min_value=0)
    no_of_weekend_nights = st.number_input('Number of weekend nights',step=1, min_value=0)
    no_of_week_nights = st.number_input('Number of week nights', step=1, min_value=0)
    type_of_meal_plan = st.slider('Type of meal plan [0 for meal plan 1, 1 for meal plan 2, 2 for meal plan 3, and 3 if meal plan was not selected]',step=1, min_value=0, max_value=3)
    required_car_parking_space = st.number_input('Whether a car parking space is required [0 for No, 1 for Yes]',step=1,min_value=0, max_value=1)
    room_type_reserved = st.slider('Type of room reserved [0 for room type 1, 1 for room_type 2, 2 fo room type 3, 3 for room type 4, 4 for room type 5, 5 for room type 6, 6 for room type 7]',step=1, min_value=0, max_value=6)
    lead_time = st.number_input('Number of days before the arrival date the booking was made.',step=1, min_value=0)
    arrival_year = st.number_input('Year of arrival', step=1, min_value=2017)
    arrival_month = st.slider('Month of arrival.',step=1, min_value=1, max_value=12)
    arrival_date = st.slider('Date of the month for arrival',step=1, min_value=1, max_value=31)
    market_segment_type = st.slider('How the booking was made [0 for Aviation, 1 for Complementary, 2 for Corporate, 3 for Offline, 4 for Online]',step=1, min_value=0, max_value=4)
    repeated_guest = st.number_input('Whether the guest has previously stayed at the hotel [0 for No, 1 for Yes]',step=1, min_value=0, max_value=1)
    no_of_previous_cancellations = st.number_input('Number of previous cancellations.',step=1, min_value=0)
    no_of_previous_bookings_not_cancelled = st.number_input('Number of previous bookings that were canceled.',step=1, min_value=0)
    avg_price_per_room = st.number_input('Average price per day of the booking.')
    no_of_special_requests = st.number_input('Count of special requests made as part of the booking.',step=1, min_value=0)
    no_of_individuals = st.number_input('The total number of individuals (adults and children)',step=1, min_value=0)
    no_of_days_booked = st.number_input('The total number of nights booked (weekend included)',step=1, min_value=0)




    user_input = [no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, type_of_meal_plan, required_car_parking_space, room_type_reserved,
                  lead_time, arrival_year, arrival_month, arrival_date, market_segment_type, repeated_guest, no_of_previous_cancellations,
                  no_of_previous_bookings_not_cancelled, avg_price_per_room, no_of_special_requests, no_of_individuals, no_of_days_booked]

    scaled_data = scaler.transform(np.array([user_input]))  #scaling the input
    if st.button('Predict'):
      prediction = model.predict(scaled_data)
      output = prediction[0]
      if output == 1:
        st.success("The booking will not be canceled")
      else:
         st.success("The booking will be canceled")
    


# Run the app
if __name__ == "__main__":
    main()
