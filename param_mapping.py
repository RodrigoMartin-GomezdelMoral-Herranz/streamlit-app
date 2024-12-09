# param_mapping.py

def map_parameters(strategy, scoring_method, method_type, score_type):
    """
    Mapea los parámetros seleccionados en la interfaz a los valores esperados por las librerías.

    Args:
        strategy (str): Estrategia de búsqueda seleccionada.
        scoring_method (str): Método de scoring seleccionado.
        method_type (str): Tipo de método para aprendizaje paramétrico.
        score_type (str): Tipo de métrica de scoring para el aprendizaje paramétrico.

    Returns:
        dict: Diccionario con los valores mapeados.
    """
    # Mapeo para la estrategia de búsqueda (structure learning)
    strategy_mapping = {
        "Peter-Clark": "pc",
        "Hill-Climbsearch": "hc",
        "NaiveBayes": "nb"
    }
    
    # Mapeo para los métodos de scoring (structure learning)
    scoring_mapping = {
        "Bayesian Information Criterion": "bic",
        "K2 metric": "k2"
    }
    
    # Mapeo para los métodos de aprendizaje paramétrico
    method_type_mapping = {
        "Maximum Likelihood Estimation": "ml",
        "Bayesian Estimation": "bayes"
    }
    
    # Mapeo para los métodos de scoring en aprendizaje paramétrico
    score_type_mapping = {
        "Bayesian Information Criterion": "bic",
        "Bayesian Dirichlet equivalent uniform prior": "bdeu"
    }
    
    return {
        "structure_strategy": strategy_mapping.get(strategy, "hc"),  # Valor por defecto: 'hc'
        "structure_scoring": scoring_mapping.get(scoring_method, "bic"),  # Valor por defecto: 'bic'
        "parameter_method": method_type_mapping.get(method_type, "ml"),  # Valor por defecto: 'MLE'
        "parameter_score": score_type_mapping.get(score_type, "bic")  # Valor por defecto: 'bic'
    }