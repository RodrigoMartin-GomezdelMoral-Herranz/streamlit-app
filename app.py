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

#basic_network_file = "net/starting_net.html"
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
    
    default_structure = bn.structure_learning.fit(
        selected_data,
        methodtype=mapped_params["structure_strategy"],
        scoretype=mapped_params["structure_scoring"]
    )
    default_structure = bn.independence_test(
        default_structure,
        selected_data,
        alpha=0.05,
        prune=False
    )
    default_model = bn.parameter_learning.fit(
        default_structure,
        selected_data,
        methodtype=mapped_params["parameter_method"],
        scoretype=mapped_params["parameter_score"]
    )
    st.session_state["model"] = default_model

    st.session_state["CPDs"] = bn.print_CPD(default_model)
    default_properties = bn.plot(default_structure)
    edge_properties = default_properties["edge_properties"]
    st.session_state["edges_df"] = pd.DataFrame([
        {"source": source, "target": target, "weight": properties["weight"]}
        for (source, target), properties in edge_properties.items()
    ])

    default_net = Network(height="600px", width="100%", directed=True)
    for index, row in st.session_state["edges_df"].iterrows():
        default_net.add_node(row["source"], label=row["source"])
        default_net.add_node(row["target"], label=row["target"])
        default_net.add_edge(row["source"], row["target"], value=row["weight"])
    default_net.save_graph(trained_network_file)

    with open(trained_network_file, "r", encoding="utf-8") as f:
        st.session_state["trained_network_html"] = f.read()
    st.session_state["trained_network"] = default_net

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

        net = Network(height="600px", width="100%", directed=True)
        for index, row in edges_df.iterrows():
            net.add_node(row["source"], label=row["source"])
            net.add_node(row["target"], label=row["target"])
            net.add_edge(row["source"], row["target"], value=row["weight"])
        
        net.save_graph(trained_network_file)

        st.session_state["trained_network"] = net
        st.session_state["CPDs"] = CPDs
        st.session_state["edges_df"] = edges_df

        if os.path.exists(trained_network_file):
            with open(trained_network_file, "r", encoding="utf-8") as f:
                trained_network_html = f.read()
            st.session_state["trained_network_html"] = trained_network_html
        else:
            st.error("Error")

        st.success("Bayesian Network Trained and Visualized Successfully!")

if st.session_state["trained_network"]:
    html(st.session_state["trained_network_html"], height=600)

    with st.expander("Conditional probabilities"):
        if st.session_state["CPDs"]:
            node = st.selectbox(
                "Select variable",
                list(st.session_state["CPDs"].keys())
            )
            if node:
                st.write(st.session_state["CPDs"][node])
        else:
            st.warning("No conditional probabilities available.")
else:
    st.warning("No trained network available. Default network loaded.")


if st.session_state["trained_network"]:
    st.header("Inference Calculator")
    st.markdown("* Use this section to infer probabilities for a selected variable given evidence.")

    target_variable = st.selectbox(
        "Target Variable",
        list(st.session_state["CPDs"].keys()),
        index=list(st.session_state["CPDs"].keys()).index("ADHERENCE_Medication")
    )

    evidence_variables = st.multiselect(
        "Evidence Variables",
        [var for var in selected_data.columns if var != target_variable]
    )
    

    evidence = {}
    if evidence_variables:
        st.write("Provide values for selected variables:")
        for var in evidence_variables:
            unique_values = selected_data[var].value_counts().index.tolist()
            evidence[var] = st.radio(f"Evidence for {var}:", unique_values)
    
    if st.button("Perform Inference"):
        if not evidence_variables:
            st.warning("No evidence variables selected. Please select at least one.")
        elif target_variable:
            try:
                query_result = bn.inference.fit(
                    st.session_state["model"],
                    variables=[target_variable],
                    evidence=evidence if evidence else None
                )
                #st.write(f"Inference result for {target_variable}:")
                st.write(query_result)
            except Exception as e:
                st.error(f"Error during inference: {e}")
        else:
            st.warning("Please select a target variable.")

st.markdown("---")
st.markdown("Second version v2.3")