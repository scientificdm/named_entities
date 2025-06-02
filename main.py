import streamlit as st
import numpy as np
import base64
import json

from io import StringIO
from gliner import GLiNER

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import spacy
from spacy.training import offsets_to_biluo_tags

stopWords = set(stopwords.words('english')) # list of stop words

nlp = spacy.load("en_core_web_sm")

st.set_page_config(
    page_title="AgroNER",
    page_icon="icon.png"
)

@st.cache_resource
def load_model():
    #return GLiNER.from_pretrained("koptelovmax/ner_finetune")
    return GLiNER.from_pretrained("koptelovmax/agroner_large")

def set_header():
    LOGO_IMAGE = "logo.png"

    st.markdown(
        """
        <style>
        .container {
            display: flex;
        }
        .logo-text {
            font-weight:700 !important;
            font-size:50px !important;
            color: #8fbb94 !important;
            padding-left: 10px !important;
        }
        .logo-img {
            float:right;
            width: 28%;
            height: 28%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
            <p class="logo-text">AgroNER <span style="color:grey;"><font size="2">named-entities in agro-waste management</font></span></p>
        </div>
        """,
        unsafe_allow_html=True
    )

def main():
    
    # Fancy header:
    set_header()
    
    # Separate tabs with file upload and hardcoded examples:
    tab1, tab2 = st.tabs(["Text input", "Examples"])
    
    fileInput = False

    with tab1:
        # Input text from a file:
        uploaded_file = st.file_uploader("File upload:")
        if uploaded_file is not None:
            # Process uploaded file:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8")) # To convert to a string based IO
            text_from_file = stringio.read() # To read file as string
            fileInput = True
        else:
            fileInput = False
           
        input_labels = st.text_input(
        "Labels:",
        "price, legislation, availability of by-products, development",
        )
                
    with tab2:
        text_labels_box = st.selectbox(
        "Select example:",
        ('Example 1: Main issues and challenges for PHA-applications',
         'Example 2: Main issues and challenges of agricultural by-product valorisation',
         'Example 3: Main issues and challenges of PHA valorisation'),
        )
        
        if text_labels_box == "Example 1: Main issues and challenges for PHA-applications":
            example_text = "Price: The customer and the end customer are not willing to pay more for these types of products\n\nThe machinery should also adapt to this new packaging, and in the future it should be able to adapt to by-products of different types of crops to avoid stock break\n\nStable supply of by-products to meet demand on an ongoing basis, without stock breaks\n\nAvailability of bioproducts suitable for the specific needs of the industry\n\nFacilitation would be needed to be able to test and buy new solutions both for agricultural use and for packaging for final consumption. This means money, time, and solid networks of collaboration between companies, technological centers, and administrations, putting in facilities and removing barriers\n\nWe find problems in collaborative work with research or technological centers for the generation of new solutions from by-products\n\nwe see a barrier to being able to find funding for trials with new materials\n\nInvestments to be able to produce or manufacture these new materials ourselves or within our local environment\n\nAnd we trust that technology and knowledge will advance, but they seem to do so very slowly, at least to our sense for practical purposes.\n\nBiodegradability (too slow or too fast) and price.\n\nIn the end, science advances, and it is possible to find good solutions, but the administration (government) does it so fast that we can encounter problems related to having a favorable legislative framework."
            example_labels = "price, legislation, availability of by-products, development"
        
        elif text_labels_box == "Example 2: Main issues and challenges of agricultural by-product valorisation":
            example_text = "In the case of the hood, which we don't have that much experience with, seasonal availability would be the most important fact, along with transportation to deal with it.\n\nWe encounter this same problem with the by-products of wine and olive oil, where we also have to add the issue of transport from the cellars or mills where the by-product is obtained to the possible treatment plant and also the seasonal availability of both crops, which means that all the by-products obtained must be processed in two months.\n\nLimited seasonal availability of feedstock / poor continuity throughout the year\n\nDue to the seasonality of olive oil production, the work of the Biosansa company does have a high degree of seasonality, which at the moment is reinforced by the need to work quickly to avoid decomposition at outdoors.\n\nAll the previous points have a certain influence, all very much conditioned by the seasonality and high variability of the productions.\n\nLimited seasonal availability of raw materials and poor continuity throughout the year.\n\nSince a constant volume is required to be able to make a very accurate forecast of product availability that will be available with a certain type of forecast, these are things that are difficult to predict in our sector.\n\nBetween August and October, they collect 250 tonnes of grape marc a day. Wine lees are collected between November and May. The extraction phase should take place at this point. Between June and August, they do not collect any raw materials, but they take advantage of this to transform some of the bases they prepare at the peak of the collection period.\n\nOur food production operation runs all year round at least five day a week and sometimes even on the weekend. Our operation is relatively constant throughout the entire year.\n\nTransport: Sufficient space is required to be able to make the collection.\n\nTransportation: Logistics and collection in particular are very expensive because farms are spread across the country.\n\nTransport: Sufficient space is required to be able to make the collection.\n\nTransportation is also limited and incurs an added cost.\n\nTransportation: Logistics and collection in particular are very expensive because farms are spread across the country.\n\nLogistics of the necessary volumes (for example, storage, regional distribution) Large volumes are generated in a short time, and the transformers sometimes cannot absorb them; therefore, there is little added value to the product.\n\nThe fermentation residues contain a relatively high amount of water. They are really wet. This is a problem for transportation as it increases the transportation costs.\n\nBut drying the fermentation residues would be incredibly energy and therefore cost intensive. It is therefore not viable to dry the residues. But long transport routes are unviable with the high water content that we currently have in our fermentation residues.\n\nLogistics of the necessary volumes (for example, storage, regional distribution) Being outside, problems can currently arise due to exposure to environmental conditions (rain and humidity).\n\nCo-products cannot really be stored on central logistics platforms to optimise logistics, because given certain uses for these raw materials (anti-oxidant principle, colouring), the raw materials need to be collected and processed quickly. In addition, such central platforms for concentrating raw materials would require major facilities that meet food standards (pest control and material deterioration, among other things).\n\nSince collection is done on demand, there are a number of aspects that must be well managed, such as volume logistics at each collection and storage point.\n\nThis is especially an issue in winter, as the farmers are not allowed to fertilise their fields during winter, but we produce potato products every day. The potato waste therefore needs to be stored during the winter months, but this is not handled internally in our company. We are working together with an external logistics company and they are handling all our wastes from the moment it leaves our biogas plant.\n\nWith regard to the challenges and difficulties, in the case of the by-products of the nuts, and specifically of the skins of almonds and hazelnuts, we are talking about a logistical issue of volumes necessary for the development to be profitable, and in the specific case of the almond skins, the need to dry beforehand makes the product more expensive before treating it, which makes the cost of treatment high. In the case of the hood, which we don't have that much experience with, seasonal availability would be the most important fact, along with transportation to deal with it.\n\nProcessing difficulties due to the chemical composition (for example, the high lignin content of grape skins). Products with a high concentration of water that make it difficult to compost the entire fraction.\n\nProcessing difficulties due to chemical composition (e.g. high lignin content of grape stalk): Because of climate change, they are not extracting the same thing from raw materials. The quality of the co-product is changing.\n\nChemical deterioration is probably also important to consider.\n\nRegional legislation presents an important limitation regarding the emission of water vapour, with very restrictive parameters in Catalonia compared to Spanish legislation, it represent a problem to re-use water from the olive oil production in agriculture.\n\nIf a major campaign to uproot vines were launched, Grap'Sud would be badly affected because it would receive much less raw material. Animal nutrition is highly regulated. Only one additive is authorised for grapes.\n\nRather the issue of recognising biogas. At the moment, biogas in natural gas quality is recognised as methane and is partly marketed, which is partly important for legislation.\n\nLegislative issues"
            example_labels = "transportation, processing difficulties, limited seasonal availability of feedstock, logistics, legislation"
            
        elif text_labels_box == "Example 3: Main issues and challenges of PHA valorisation":
            example_text = "The fermentation process can also not be scaled up infinite. There is a maximum size that you will reach dictated by physics.\n\nSo, you will need to achieve economy of scale but will be limited in the fermenter size. You need to reach volumes of 50’000 to 100’000 tons per year.\n\nNo manufacturer has managed to build a 50’000 tonne plant until today… globally…\n\nKANEKA have been researching on PHAs since the 1980s… That’s over 40 years of research and development! They have been going through their ups and downs and they are still far away from having a 50’000 ton production plant.\n\nAt least five to seven years after you have a pilot plant with a volume of 1000 tons per year. KANEKA had a pilot plant with 2k tonnes volume per year and is now taking the first step towards commercial size production. This means that they had to pay for each tonne they sold beforehand. During the pilot phase you are not able to sell at the same price level as your production costs. But you need to sell, and you need customer. So, for each tonne you produce you need to invest money. So, you really have to have a lot of staying power. You need to be able to spend for several years before ever seeing any revenue.\n\nBioreactors can be quite big. Of-course there are technical borders for the size of a bioreactor. But it is possible to increase the number of bioreactors if the size limit is met. So, the process itself can be technically solved. It is possible to produce PHAs on a large scale. We can see that in China. You just need to increase the number of bioreactors.\n\nIf money was not a limiting factor 5 to 10 years.\n\nThere are PHA clips available today, but they are too expensive. Most clips today are made from PP or bioplastic but are only biodegradable in industrial compost. This also means that farmers have to remove them by hand, one by one, from the plantations.\n\nThere has been a shampoo bottle on the market for example that was 10 times more expensive than conventional shampoo bottles around 40 years ago. As they were much harder to process, they were not able to establish on the market with the shampoo bottles.\n\nAnd nowadays the processes are still too expensive. The raw materials actually only make up for around 50 % of the whole costs.\n\nPHAs are by far the most expensive biopolymer on the market. They will easily cost 5 € or more and can therefore simply not compete with conventional polymers or other biopolymers.\n\nDepending on the polymer you’d have to pay between 1 € or 1,5 € per kg, but PHAs cost over 5 € /kg and nothing will change about it in the future.\n\nYou need a couple of 100 million Euros to successfully manage upscaling.\n\nPrice\n\nThe fermentation process to PHAs is rather difficult.\n\nBut mastering the fermentation process is very difficult. These companies all work with sugar or vegetable oils as raw materials for bacteria. No one works with residues. The fermentation is a very complex process and requires very uniform raw materials with predefined properties. The polymer is enclosed in the bacteria during the fermentation process and must be extracted. To do that the biomass of the bacteria must be removed.\n\nThe problem with using residues is that they cannot be used for the fermentation process as they are. You need extracted sugars or oils or starch for the process to feed the bacteria.\n\nThe residue without processing cannot be used as a raw material for fermentation to PHAs because it is not concentrated enough. You need around 60 % sugar in your raw material. If you only have 10 % sugar in your solution it is not even necessary to try fermentation.\n\nAnd to be honest, the whole fermentation process is really not the best solution ecologically it needs a lot of energy to produce fermented PHAs\n\nAny outdoor application that needs to be durable is not a viable option for PHAs. PHAs degrade too easily. They break down really quickly. As soon as they get in contact with soil or water the hydrolysis starts, or the bacteria and fungi start to break the PHAs down.\n\nPHAs degrade very quickly. They get attacked by all organisms and can degrade within days or weeks. They lose their properties really quick if they get in touch with soil. PHAs have been around for billions of years, they are known to every fungus and bacterium. That’s the reason why they can break them down very efficiently. There is always an organism who can break them down. It will degrade very quickly when in the soil. It will degrade within a few days or weeks.\n\nAnd using PHAs in agricultural or horticultural applications is really difficult. You cannot use biopolymer for silage as the bacteria would simply munch away on the foil and then again you wouldn’t want to feed the silage foil to your cattle.\n\ntime frame of degradation corresponding to climate zone\n\nTime frame of degradation corresponding to climate zone and flexibility (yes or no)\n\nBut storage over long time takes up lots of space and is therefore costly. The sugar industry has managed to produce in in campaigns. This has become standard among the industry. So, the industry is aware of that storage problem and knows how to deal with it. This is a solvable problem. In Greece there are for example different varieties of oranges this enables farmers to harvest all year round. Continuous harvest throughout the year will enable a continuous supply of orange peel. This is probably similar with greenhouse culture e.g., tomato in Spain or Italy,\n\nYes, for sure. The raw material must be transport worthy (highly concentrated).\n\nIt is a different story with feedstock. That would be the main limit in my opinion."
            example_labels = "scalability, price, production, durability, availability of feedstock"
    
    # Load fine-tuned model:
    model = load_model()
    
    col_1, col_2 = st.columns([1,2], vertical_alignment="top")
    with col_1:
        # Select considering titles:
        threshold_flag = st.checkbox("Use optimal threshold (optimal number of entities)", value=True)
    with col_2:
        # Select threshold:
        if threshold_flag:
            #threshold_input = st.slider("Select a threshold (the lower the threshold, the more entities will be discovered, and vice versa):", min_value=0.1, max_value=0.9, value=0.5, step=0.01, disabled=True)
            threshold_input = st.slider("Select a threshold (the lower the threshold, the more entities will be discovered, and vice versa):", min_value=0.1, max_value=0.9, value=0.6, step=0.01, disabled=True)
        else:
            #threshold_input = st.slider("Select a threshold (the lower the threshold, the more entities will be discovered, and vice versa):", min_value=0.1, max_value=0.9, value=0.5, step=0.01, disabled=False)
            threshold_input = st.slider("Select a threshold (the lower the threshold, the more entities will be discovered, and vice versa):", min_value=0.1, max_value=0.9, value=0.6, step=0.01, disabled=False)
           
    if st.button('Submit'):
        # Determine which text input to use:
        if fileInput:
            text_to_process = text_from_file
            text_labels = input_labels
            fileInput = False
        else:
            text_to_process = example_text
            text_labels = example_labels
        
        # Separate text labels:
        entities = [i.strip() for i in text_labels.split(",")]
        
        # Colors used for visualization of labels:
        colors = ["red","blue","green","violet","orange","grey","rainbow"]
        label_dic = {}

        # Separate input text by paragraphs and words:
        text_lines = text_to_process.split('\n\n')
        
        labels = []
        for j in range(len(entities)):
            labels.append(entities[j])
            #color_dic[entities[j].split(" ")[0].upper()] = colors[j]
            label_dic[entities[j].split(" ")[0].upper()] = j
  
            # Output dictionary initialization:
            entity_type_list = entities[j].split(" ")
            if len(entity_type_list) > 1:
                entity_type_list = [w.lower() for w in entities[j].split(" ") if not w.lower() in stopWords] # remove stop words
        
        y_pred = []
        tokens = []
        export_data = {}
        for l in range(len(text_lines)):
            # Make prediction:
            if threshold_flag:
                #pred_entities = model.predict_entities(text_lines[l], labels, threshold=0.5)
                pred_entities = model.predict_entities(text_lines[l], labels, threshold=0.6)
            else:
                pred_entities = model.predict_entities(text_lines[l], labels, threshold=threshold_input)
            
            # Convert results of Gliner to spaCy format, e.g. [(7, 13, "LOC")], etc.
            pred_entities_spacy = []
            for item in pred_entities:
                pred_entities_spacy.append((item['start'],item['end'],item['label'].split(" ")[0].upper()))
        
            # Convert results to the BILOU format using spaCy, e.g. ["O", "O", "U-LOC", "O"], etc.
            doc = nlp(text_lines[l])
            tokens.append([token.text for token in doc])
            y_pred.append(offsets_to_biluo_tags(doc, pred_entities_spacy))
            
            # Prepare results for export to JSON format:
            for item in pred_entities:
                dict_tmp = {}
                dict_tmp["label"] = item['label']
                dict_tmp["entity"] = item['text']
                    
                if "pred_entities" not in export_data.keys():
                    export_data["pred_entities"] = [dict_tmp]
                else:
                    export_data["pred_entities"].append(dict_tmp)
        
        # Convert highlighted text using BILOU format + compute frequencies:
        entity_frequency = np.zeros((len(entities),len(text_lines)),int) # frequencies of predicted entities
        
        result_to_output = ""
        if len(entities) <= 7:
            for i in range(len(y_pred)):
                for j in range(len(y_pred[i])):
                    if "U-" in y_pred[i][j]:
                        entity_idx = label_dic[y_pred[i][j].split("U-")[1]]
                        #result_to_output += ":"+colors[entity_idx]+"-background["+tokens[i][j]+"] "
                        if j < len(tokens[i])-1:
                            if tokens[i][j+1] not in '!)*,.:;-?]}':
                                if tokens[i][j] not in '-([{':
                                    result_to_output += ":"+colors[entity_idx]+"-background["+tokens[i][j]+"] "
                                else:
                                    result_to_output += ":"+colors[entity_idx]+"-background["+tokens[i][j]+"]"
                            else:
                                result_to_output += ":"+colors[entity_idx]+"-background["+tokens[i][j]+"]"
                        else:
                            result_to_output += ":"+colors[entity_idx]+"-background["+tokens[i][j]+"]"
                        entity_frequency[entity_idx][i]+=1
                    elif "B-" in y_pred[i][j]:
                        entity_idx = label_dic[y_pred[i][j].split("B-")[1]]
                        #result_to_output += ":"+colors[entity_idx]+"-background["+tokens[i][j]+" "
                        if tokens[i][j+1] not in '!)*,.:;-?]}':
                            if tokens[i][j] not in '-([{':
                                result_to_output += ":"+colors[entity_idx]+"-background["+tokens[i][j]+" "
                            else:
                                result_to_output += ":"+colors[entity_idx]+"-background["+tokens[i][j]
                        else:
                            result_to_output += ":"+colors[entity_idx]+"-background["+tokens[i][j]
                        entity_frequency[entity_idx][i]+=1
                    elif "L-" in y_pred[i][j]:
                        result_to_output += tokens[i][j]+"] "
                    else:
                        if j < len(tokens[i])-1:
                            if tokens[i][j+1] not in '!)*,.:;-?]}':
                                if tokens[i][j] not in '-([{':
                                    result_to_output += tokens[i][j] + " "
                                else:
                                    result_to_output += tokens[i][j]
                            else:
                                result_to_output += tokens[i][j]
                        else:
                            result_to_output += tokens[i][j]
                result_to_output += "\n\n"
        
        # Display legend with colors
        text_to_output = ""
        st.markdown("**Desired entities:**")
        for i in range(len(entities)-1):
            text_to_output += ":"+colors[i]+"-background["+entities[i]+"], "
        if len(entities) != 0:
            text_to_output += ":"+colors[i+1]+"-background["+entities[i+1]+"]."
        if text_to_output != "":
            st.markdown(text_to_output)

        # Ouput prediction results:        
        st.markdown("**Predicted entities:**")
        if result_to_output != "":
            st.markdown(result_to_output)
        else:
            if len(entities) > 7:
                st.write("Too many labels! Please decrease the number by a maximum of 7 labels and resubmit the query.")
            else:
                st.write("Something went wrong!")
        
        # Compute entity counts by their type:
        freq_segm_list = []
        freq_total_list = []
        for i in range(len(entities)):
            freq_segm_list.append((np.sum([bool(j) for j in entity_frequency[i]]),entities[i],colors[i]))
            freq_total_list.append((np.sum(entity_frequency[i]),entities[i],colors[i]))
        
        # Sort frequencies by descending order:
        freq_segm_sorted = sorted(freq_segm_list, key=lambda tup: tup[0], reverse=True)
        freq_total_sorted = sorted(freq_total_list, key=lambda tup: tup[0], reverse=True)
        
        # Output the results:        
        text_to_output = "**Total (segments):**"
        for i in range(len(entities)-1):
            text_to_output += ":"+freq_segm_sorted[i][2]+"-background["+freq_segm_sorted[i][1]+"] " + str(freq_segm_sorted[i][0]) + ", "
        if len(entities) != 0:
            text_to_output += ":"+freq_segm_sorted[i+1][2]+"-background["+freq_segm_sorted[i+1][1]+"] " + str(freq_segm_sorted[i+1][0])# + "."
        st.markdown(text_to_output)
        
        text_to_output = "**Total entities:**"
        for i in range(len(entities)-1):
            text_to_output += ":"+freq_total_sorted[i][2]+"-background["+freq_total_sorted[i][1]+"] " + str(freq_total_sorted[i][0]) + ", "
        if len(entities) != 0:
            text_to_output += ":"+freq_total_sorted[i+1][2]+"-background["+freq_total_sorted[i+1][1]+"] " + str(freq_total_sorted[i+1][0])# + "."
        st.markdown(text_to_output)
               
        # Convert predicted entities to JSON:
        json_string = json.dumps(export_data)
        
        # Export data to a separate file:
        st.download_button(
            label = "Export predicted entites to JSON",
            file_name="output.jsonl",
            mime="application/jsonl",
            data=json_string,
        )
        
if __name__ == "__main__":
    main()