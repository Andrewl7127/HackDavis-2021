# TrackTogether

https://youtu.be/-CYTji5Ea9w

In order to ensure information about our on-going pandemic is inclusive and accessible to everyone regardless of ability or disability, we have created a web application that uses **speech recognition, narration, deep learning, and various other technical tools** to present the latest information about COVID-19 to users. In the last year, there have been a variety of applications developed to spread verified COVID-related information to users. Despite this, we realized a gap in catering for those with visual impairments. We decided to address this issue with the tremendous capabilities of **speech recognition technology to essentially create a virtual assistant for COVID-19 information** that can perform a variety of tasks ranging from answering questions about the virus using an AI-powered chatbot, to returning the top-5 global COVID-19 headlines for the current day, to even enabling quick-and-easy dissemination of COVID-19 county-specific updates to family and friends without even having to touch your phone. All of our information is continuously updated. We wanted to **empower people with visual disabilities** to ultimately make more informed decisions about COVID-19 for their own health and safety. 

By educating users, and reducing information access inequalities, our application addresses goals #4 and #10 of the 17 UN Sustainable Development Goals.  Our team also consisted of members from 2 different time-zones (Hong Kong Time, and Pacific Standard Time), making this a **global hack!**  

## Speech Recognition Technology

As mentioned above, our app focuses on enabling the visually impaired. In order to cater to this specific demographic, we needed to deviate from traditional text input/output and transition to a speech to text/text to speech functionality. To do this, we utilized the PyAudio and SpeechRecognition libraries for speech to text and the Pyttsx3 library for text to speech. By using speech recognition, **we allow visually impaired users to interact with the app without any use of eyesight.** Users can simply verbally select from our list of options and the bot will read aloud the outputs. Additionally, our app pairs seamlessly with the text to speech capabilities of mobile devices with programs such as Siri that can read aloud texts.

## An AI-Powered Chatbot

Apart from using speech recognition technology to improve the accessibility of information, TrackTogether also makes use of the incredible capabilities of deep learning to create an **AI-powered chatbot** that answers COVID-related questions.  By using the Tensorflow and the TF-Learn modules in python, we created, trained, and deployed a deep-learning neural network that helps us predict the type of questions asked by a user. Our model architecture consists of three hidden layers in total. Specifically, this is made up of one pre-trained word embedding layer that accounts for the similarity of words used, and two densely connected layers consisting of 8 and 16 neurons respectively. We also used the NLTK library to preprocess, lemmatize, and tokenize the text used. With a **96%** accuracy, your model would take in the user's input and return the topic associated with their input. Using this predicted topic, we would randomly pick from a handful of responses and return these to the user. This AI-powered chatbot, which we created from scratch, was integrated with our speech recognition functionality to **enable visually impaired individuals to easily ask any COVID-19 related questions, and receive an accurate, and informative response**.   

## Usage of Twilio’s API

Twilio’s API allows users to programmatically send SMS messages to different people. Using Twilio we allow the user to send SMS messages containing the new/cumulative COVID-19 cases/deaths in their county. These messages are programmed to be sent every 24 hours and are updated daily by querying from the  New York Times’ COVID-19 GitHub repository. Furthermore, messages can be halted by the user when the keyword STOP is entered.

## Real-time COVID news updates

In order to get an idea of the most popular COVID articles at the moment, we web-scraped the hottest posts on the Coronavirus subreddit. These posts are exclusively news feed/news articles pertaining to the COVID-19 pandemic and are updated every day based on popularity/interest. This information is conveyed on our Streamlit app and at the user’s discretion can be sent via SMS to their phone.

## Data and Visualisations 

In order to get a scope of the scale of COVID-19 we additionally created a live, interactive dashboard in order to get a scale of COVID-19’s impact on a county-by-county level. On top of the bot, we created a dashboard to have everything in one place and illustrate some unique metrics. In our [heroku app](https://tracktogether.herokuapp.com/) **(takes 5 seconds to load)** we illustrate multiple county-level metrics. To start off we look at the cumulative COVID cases by county (updated in real-time) to identify hotspots across the nation. We also use a novel clustering algorithm built to cluster different counties across the United States by user-specified metrics. The results of clustering are visualized and can be downloaded for further analysis/research by a user in CSV form. The beauty of this approach is that the user chooses what metrics they care about (i.e. ICU Beds by county, COVID Cases per capita, etc.) and can see how different counties are categorized/ranked.

## Conclusion

This pandemic has demonstrated the deadly impacts of misinformation in society. Our application ensures that visually impaired individuals can **easily access verified information** about the pandemic so that they can make as **informed decisions** about their health and safety as the rest of our society. While our bot application is restricted to our local devices, we hope to expand the offerings of our app and enhance the capabilities of our chatbot in the future. We also hope to make our app public and deploy it on an easily accessible platform such as a website or mobile app.

