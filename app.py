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
    ["Hill-Climbsearch", "NaiveBayes"]
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
    ["Maximum Likelihood Estimation"]
)
score_type = st.sidebar.selectbox(
    "Scoring methods",
    ["Bayesian Information Criterion", "Bayesian Dirichlet equivalent uniform prior"]
)

mapped_params = map_parameters(strategy, scoring_method, method_type, score_type)

st.header("Training and Visualization")

basic_network_file = "net/net.html"
trained_network_file = "net/bayesian_network.html"

#keys = None
#CPDs = None

#if not os.path.exists(trained_network_file):
#    if os.path.exists(basic_network_file):
#        with open(basic_network_file, "r", encoding="utf-8") as f:
#            basic_network_html = f.read()
#        html(basic_network_html, height=600)

if "trained_network" not in st.session_state:
    st.session_state["trained_network"] = None
    st.session_state["CPDs"] = None
    st.session_state["edges_df"] = None

if st.sidebar.button("Train Bayesian Network"):
    with st.spinner("Training Bayesian Network..."):
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

        st.session_state["model"] = model

        CPDs = bn.print_CPD(model)
        keys = list(CPDs.keys())

        properties = bn.plot(structure)
        edge_properties = properties["edge_properties"]

        edges_df = pd.DataFrame([
            {"source": source, "target": target, "weight": properties["weight"]}
            for (source, target), properties in edge_properties.items()
        ])

        # Crear y guardar la red con Pyvis
        net = Network(height="600px", width="100%", directed=True)
        for index, row in edges_df.iterrows():
            net.add_node(row["source"], label=row["source"])
            net.add_node(row["target"], label=row["target"])
            net.add_edge(row["source"], row["target"], value=row["weight"])
        
        net.save_graph(trained_network_file)

        # Guardar el estado de la red en la sesión
        st.session_state["trained_network"] = net
        st.session_state["CPDs"] = CPDs
        st.session_state["edges_df"] = edges_df

        if os.path.exists(trained_network_file):
            with open(trained_network_file, "r", encoding="utf-8") as f:
                trained_network_html = f.read()
            st.session_state["trained_network_html"] = trained_network_html
        else:
            st.error("Error al cargar la red entrenada. Asegúrate de que se haya generado correctamente.")

        st.success("Bayesian Network Trained and Visualized Successfully!")

if st.session_state["trained_network"]:
    html(st.session_state["trained_network_html"], height=600)

    with st.expander("See the conditional probabilities"):
        node = st.selectbox(
            "Select node",
            list(st.session_state["CPDs"].keys())
        )
        if node is not None:
            st.write(st.session_state["CPDs"][node])

if st.session_state["trained_network"]:
    #st.markdown("---")
    st.header("Inference Calculator")
    st.markdown("* Use this section to infer probabilities for a selected variable given evidence.")

    target_variable = st.selectbox(
        "Select the variable to infer:",
        list(selected_data.columns)
    )

    st.markdown("### Evidence Selection")
    evidence = {}
    for col in selected_data.columns:
        if col != target_variable:
            unique_values = selected_data[col].value_counts().index.tolist()
            evidence[col] = st.selectbox(
                f"Select value for {col} (leave empty for no evidence):",
                options=[""] + unique_values,
                format_func=lambda x: "No Evidence" if x == "" else str(x)
            )

    evidence = {k: v for k, v in evidence.items() if v != ""}

    if st.button("Infer"):
        if evidence:
            model = st.session_state.get("model")
            try:
                result = bn.inference.fit(
                    model,
                    variables=[target_variable],
                    evidence=evidence
                )
                st.success(f"Inference successful for {target_variable}!")
                st.write(result)
            except Exception as e:
                st.error(f"Error during inference: {e}")
        else:
            st.warning("Please provide at least one piece of evidence to infer probabilities.")

st.markdown("---")
st.markdown("Second version v2")