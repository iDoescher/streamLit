import pandas as pd
import streamlit as st
import numpy as np
from openai import OpenAI

apiKey = st.text_input("Enter your key:")
client = OpenAI(
  api_key= apiKey
)

st.set_page_config(layout="wide")

st.title("Recommendation App Beta")

s= st.file_uploader("Upload Service File Here")
c = st.file_uploader("Upload Controls File Here")
f = st.file_uploader("Upload FOB File Here")



if s is not None and c is not None and f is not None:

    serviceDf = pd.read_excel(s,header=1)
    fobDf = pd.read_excel(f, header = 1)
    controlsDf = pd.read_excel(c, header = 1)

    #Format Data
    serviceDf = serviceDf.drop(serviceDf.index[-1])
    fobDf = fobDf.drop(fobDf.index[-1])
    controlsDf = controlsDf.drop(controlsDf.index[-1])

    #Merged Dataframes on Location
    merge1 = pd.merge(serviceDf, fobDf, on='Loc', how='inner')
    merged = pd.merge(merge1, controlsDf, on = 'Loc', how = "inner")





    st.text("Data Uploaded:")
    merged = merged.set_index("Loc")
    merged
    #Col 2: OEPE W/O Parked (=> 90 && 150 <=)
    #Col 3: R2P (>=50 && 130 <= )
    #Col 4: FOB % (>= 2.3 && 3.5 <=)
    #Col 5: POS Overrings (30 <)
    #Col 6: Cash Refund Amt (50 <= )
    #Col 7: Cashless Refund Amt (NA)
    #Col 8: Actual Labor % (.5 < )



    #Create New Copy
    nDf = merged.copy()


    minOepe = 30 #This is the min of the dataset of OEPE
    maxOepe = 180 #This is the max of the dataset of OEPE
    goalOepe = 120 #This is the Target Goal of OEPE

    #Create conditions for np vectorization instead of for loop
    conditionsOepe = [
        (nDf['OEPE W/O Parked'] <= 90), #These are the lower bounds/troublesome areas
        (nDf['OEPE W/O Parked'] >= 91)
        #((nDf['OEPE W/O Parked'] > 90) & (nDf['OEPE W/O Parked'] < 150))
    ]
    choicesOepe = [(minOepe - nDf['OEPE W/O Parked'])/(maxOepe - minOepe), 
                (nDf['OEPE W/O Parked']-minOepe)/(maxOepe - minOepe)
                #(abs(nDf['OEPE W/O Parked'])-goalOepe)/(maxOepe - minOepe)
    ]

    nDf['OEPE Score'] = np.select(conditionsOepe, choicesOepe, default='Unknown')



    minR2P = 20 #Min of dataset of R2P
    maxR2P = 150 #Max of dataset of R2P
    goalR2P = 110 #Goal of R2P

    conditionsR2P = [
        (nDf['R2P'] <= 50), #Lower bounds as defined by managers
        (nDf['R2P'] >= 51)
        #((nDf['R2P'] > 50) & (nDf['R2P'] < 130))
    ]

    choicesR2P =  [(minR2P - nDf['R2P'])/(maxR2P - minR2P), 
                (nDf['R2P']-minR2P)/(maxR2P - minR2P)
                #(abs(goalR2P-nDf['R2P']))/(maxR2P - minR2P)
    ]

    nDf['R2P Score'] = np.select(conditionsR2P, choicesR2P)



    minFOB = 1   #Dataset min of FOB
    maxFOB = 5   #Dataset max of FOB
    goalFOB= 3    #Goal of FOB


    conditionsFOB = [
        (nDf['FOB %'] <= 2.3), #Metrics as defined by district managers
        (nDf['FOB %'] >= 3.5),
        ((nDf['FOB %'] > 2.3) & (nDf['FOB %'] < 3.5))
    ]

    choicesFOB =  [(minFOB - nDf['FOB %'])/(maxFOB - minFOB), 
                (nDf['FOB %']-minFOB)/(maxFOB - minFOB), 
                (abs(goalFOB-nDf['FOB %']))/(maxFOB - minFOB)
    ]

    nDf['FOB Score'] = np.select(conditionsFOB, choicesFOB)


    maxPosOverring = 60
    nDf['POS Overrings Score'] = (nDf['POS Overrings Amt'] - 0) / (maxPosOverring-0)

    maxCashRefund = 70
    nDf['Cash Refund Score'] = (nDf['Cash Refund Amt'] - 0) / (maxCashRefund-0)

    maxLabor = .7
    nDf['Actual Labor Score'] = (nDf['Actual Labor %'] - 0) / (maxLabor-0)



    computedMetricList = ['OEPE Score',
                        'R2P Score',
                        'FOB Score',
                        'POS Overrings Score',
                        'Cash Refund Score',
                        'Actual Labor Score']


    computedMetricDf = nDf[computedMetricList].copy()
    computedMetricDf = computedMetricDf.apply(pd.to_numeric, errors='coerce')

    computedMetricDf['Score'] = (
        (computedMetricDf['OEPE Score'] * .2)
        + (computedMetricDf['R2P Score'] * .2)
        + (computedMetricDf['FOB Score'] * .3)
        + (computedMetricDf['POS Overrings Score'] * .1)
        + (computedMetricDf['Cash Refund Score'] * .1)
        + (computedMetricDf['Actual Labor Score'] * .1)
    )

    computedMetricDf

    storeToVisit = computedMetricDf['Score'].idxmax()

    if st.button("Analyze Recommendations"):
        with st.spinner("Processing..."):
            best_row = merged.loc[storeToVisit]
            store_name = storeToVisit

            # Construct explanation prompt
            prompt = f"""
            We are evaluating McDonald's store performance based on several KPIs.
            Each store has a weighted composite score that computes the worst performing store.

            The worst performing selected store to visit is: **{store_name}**, 
            with a total score of {computedMetricDf['Score']}.

            Here are its key metrics:
            - OEPE Metric (Drive Thru Times) (Standard is around 140, higher is worse): {best_row['OEPE W/O Parked']}
            - R2P Metric (In Cafe Reciept to Getting Food Time) (standard is around 50, higher is worse): {best_row['R2P']}
            - FOB Metric (Food Over Base)(Lower is better)(Standard is around 1%): {best_row['FOB %']}
            - POS Metric (POS Cash Register Overrings)(Lower is better)(Is a problem if over 15, normal if lower): {best_row['POS Overrings Amt']}
            - Cash Metric (Cash Refund Ammounts)(Lower is better)(Is a problem if over 30, normal if lower): {best_row['Cash Refund Amt']}
            - Actual Labor Metric (Labor needed)(Lower is better, standard is around 1.0, normal if lower): {best_row['Actual Labor %']}

            Explain in simple, clear language why this store should be visited, 
            what factors contributed the most,
            what steps to take next but keep this section brief,
            Cite specific metrics to justify your reasoning.
            """

            # Send to GPT
            response = client.chat.completions.create(
                model="gpt-5",  #gpt-5 or "gpt-5-mini" for faster/cheaper
                messages=[
                    {"role": "system", "content": "You are an analytics assistant that explains data-driven recommendations clearly."},
                    {"role": "user", "content": prompt}
                ]
            )
            ranAnalysis = (response.choices[0].message.content)
        st.success(ranAnalysis)

