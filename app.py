import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import bnlearn as bn
from param_mapping import map_parameters
#from d3blocks import D3Blocks
import os
from pyvis.network import Network

st.set_page_config(page_title="Bayesian Network for Adherence to Treatment", layout="wide")

st.title("Bayesian Network Generator")

st.markdown("""
 * Use the menu at left to select data and set net parameters
 * The subsequent results will appear below
""")

data = pd.read_csv("ordered_preprocessed_data.csv")
max_features = len(data.columns) - 1

st.sidebar.header("Structure learning")

strategy = st.sidebar.selectbox(
    "Search strategy",
    ["Peter-Clark", "Hill-Climbsearch", "NaiveBayes"]
)

scoring_method = st.sidebar.selectbox(
    "Scoring methods",
    ["Bayesian Information Criterion", "K2 metric"]
)

num_features = st.sidebar.slider(
    "Number of features",
    min_value=2, max_value=max_features, value=2
)

selected_data = data.iloc[:, :num_features]

st.sidebar.header("Learning parameters")
method_type = st.sidebar.selectbox(
    "Method type",
    ["Maximum Likelihood Estimation", "Bayesian Estimation"]
)
score_type = st.sidebar.selectbox(
    "Scoring methods",
    ["Bayesian Information Criterion", "Bayesian Dirichlet equivalent uniform prior"]
)

mapped_params = map_parameters(strategy, scoring_method, method_type, score_type)

st.header("Training and Visualization")

basic_network_file = "net/net.html"
trained_network_file = "net/bayesian_network.html"

#if not os.path.exists(trained_network_file):
#    if os.path.exists(basic_network_file):
#        with open(basic_network_file, "r", encoding="utf-8") as f:
#            basic_network_html = f.read()
#        html(basic_network_html, height=600)

if st.sidebar.button("Train Bayesian Network"):
    with st.spinner("Training Bayesian Network..."):
        try:
            if not mapped_params["structure_strategy"] == "nb":
                structure = bn.structure_learning.fit(
                    selected_data,
                    methodtype=mapped_params["structure_strategy"],
                    scoretype=mapped_params["structure_scoring"]
                )
            else:
                structure = bn.structure_learning.fit(
                    selected_data,
                    methodtype=mapped_params["structure_strategy"],
                    scoretype=mapped_params["structure_scoring"],
                    root_node="ADHERENCE_Medication"
                )
            structure = bn.independence_test(
                structure,
                data,
                alpha=0.05,
                prune=False
            )
            model = bn.parameter_learning.fit(
                structure,
                selected_data,
                methodtype=mapped_params["parameter_method"],
                scoretype=mapped_params["parameter_score"]
            )

            properties = bn.plot(structure)
            edge_properties = edge = properties['edge_properties']

            edges_df = pd.DataFrame([
                {"source": source, "target": target, "weight": properties["weight"]}
                for (source, target), properties in edge_properties.items()
            ])

            #output_dir = "net"
            #if not os.path.exists(output_dir):
            #    os.makedirs(output_dir)

            # Bayesian network con pyvis  
            net = Network(height="600px", width="100%", directed=True)

            for index, row in edges_df.iterrows():
                net.add_node(row["source"], label=row["source"])
                net.add_node(row["target"], label=row["target"])
                net.add_edge(row["source"], row["target"], value=row["weight"])

            net.save_graph(trained_network_file)

            
            # Bayesian network con D3Block
            #d3 = D3Blocks()
            #output_file = os.path.join(output_dir, "bayesian_network.html")
            #d3.d3graph(edges_df, figsize=(800, 600), filepath=output_file, showfig=False)
            
            #d3.D3graph.set_edge_properties(directed=True, marker_end='arrow')

            if os.path.exists(trained_network_file):
                with open(trained_network_file, "r", encoding="utf-8") as f:
                    trained_network_html = f.read()
                html(trained_network_html, height=600)
            else:
                st.error("Error al cargar la red entrenada. Asegúrate de que se haya generado correctamente.")
            
            st.success("Bayesian Network Trained and Visualized Successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

st.markdown("---")
st.markdown("First version v1")