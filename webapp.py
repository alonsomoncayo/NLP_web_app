import pandas as pd
import numpy as np
import spacy
import streamlit as st
from streamlit_tags import st_tags
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import altair as alt
import Model

st.set_page_config(layout="wide")

info = pd.DataFrame(
    {
        "Stars": ["5", "4", "3",'2','1'],
        "Color": ["Blue", "Purple", "Pink",'Orange','Yellow']
    }
)

st.title('Build your own restaurant')

@st.cache_resource
def loading_models():
    model = Model.NLP_model()
    return model

model = loading_models()
df = model.df
top10 = model.top10
top10words = model.top10words.sort_values('Freq',ascending=False)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Restaurants distribution", "Top rated restaurants",'Define the category', "Build your menu",'Review analyzer'])

with tab1:
    st.header('Restaurants in Metropolitan area of Philadelphia')
    st.map(df,latitude='latitude',longitude='longitude',color='color')
    col1, col2,col3 = st.columns(3)
    with col1:
        st.data_editor(
        info,
        column_config={
            "Stars": st.column_config.NumberColumn(format="%d ⭐")
        },
        disabled=["Stars", "Color"],
        hide_index=True)
    with col2:
        st.markdown('''
                    ### Number of restaurants:
                    #### 16,049''')
    with col3:
        st.markdown('''
                    ### Number of reviews:
                    #### 1,210,032''')

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(top10[['name','mean','categories']],hide_index=True,width=700)
    with col2:
        st.image('https://github.com/alonsomoncayo/Project-2/blob/main/Categories.png?raw=true', caption='Top rated categories')
    st.subheader('What people say about this restaurants/categories?')
    st.bar_chart(top10words,x='Word',y='Freq')

with tab3:
    st.header("Let's define the theme for your restaurant")
    keywords = st_tags(
    label='Enter Keywords that you want to use to define your restaurant:',
    text='Press enter to add more')
    col1, col2 = st.columns(2)
    if len(keywords) > 0:
        with col1:
            tokens = ', '.join(keywords)
            topics_recomendations = model.lda_description(tokens)
            st.markdown('### You should focus your restaurant:')
            for i in topics_recomendations:
                st.markdown(f'''
                                # {int(round(i[1]*100,0))}% on {i[0]}''')
        with col2:
            df_recomendations = pd.DataFrame(topics_recomendations,columns = ['Category','Proportion'])
            c = (alt.Chart(df_recomendations).mark_arc().encode(
            theta="Proportion",
            color="Category"
            ))
            st.subheader('Proportion of categories')
            st.altair_chart(c, use_container_width=True)

        st.divider()
        word_algebra = model.word_algebra(keywords,topn=5)
        st.markdown('### Some related terms to create your restaurant based on your keywords is:')
        for i in word_algebra:
            st.markdown(f'## {i}')



with tab4:
    st.header('Use one meal to start creating your menu')
    token = st.text_input('Write one word or two separetaded by underesore (e.g. taco or ice_cream)')
    if st.button('Start building'):
        st.write(f'For {token} people is also looking for:')
        s = model.get_related_terms(token)
        d = {}
        for a, x in s.values:
            d[a] = x
        wc = WordCloud( width = 950, height = 450).generate_from_frequencies(d).to_array()
        st.image(wc)

with tab5:
    st.header('Analyze the reviews of your customers')
    col1, col2 = st.columns(2)
    with col1:
        reviews = st_tags(
        label='Enter the reviews of your clients:',
        text='Press enter to add more')
    with col2:
        if len(reviews) > 0:
            overall_rating = model.sentiment_analysis(reviews,'String')
            if overall_rating[1][0][1] > overall_rating[1][0][0]:
                prediction = 'POSITIVE'
                color = 'green'
                prob = overall_rating[1][0][1]
            else:
                prediction = 'NEGATIVE'
                color = 'red'
                prob = overall_rating[1][0][0]
            st.markdown(f'''
                        ## Overall, your custumers have a {int(prob*100)}% chance of having a :{color}[{prediction}] opinion of your restaurant''')
            
            st.divider()

            df_reviews = pd.DataFrame(reviews,columns=['Reviews'])
            individual_rating = model.sentiment_analysis(df_reviews,'DataFrame')
            individual_pred = individual_rating[0]
            individual_prob = individual_rating[1]
            uniques, counts = np.unique(individual_pred, return_counts=True)
            percentages = dict(zip(uniques, counts * 100 / len(individual_pred)))
            reviews_proportion = pd.DataFrame.from_dict(percentages,orient='index').reset_index()
            reviews_proportion.columns = ['Prediction', 'Proportion']
            reviews_proportion['Review_Prediction'] = reviews_proportion['Prediction'].apply(lambda x: 'Negative' if x == 0 else 'Posivite')
            reviews_proportion.drop('Prediction',inplace=True,axis=1)
            c_rev = (alt.Chart(reviews_proportion).mark_arc().encode(
            theta="Proportion",
            color="Review_Prediction"
            ))
            st.subheader('Proportion of reviews')
            st.altair_chart(c_rev, use_container_width=True)