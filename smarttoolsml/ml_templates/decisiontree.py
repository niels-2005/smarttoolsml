from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz


def save_tree_as_png(model):
    dot_data = export_graphviz(
        model,
        filled=True,
        rounded=True,
        class_names=["Setosa", "Versicolor", "Virginica"],
        feature_names=["petal length", "petal width"],
        out_file=None,
    )
    graph = graph_from_dot_data(dot_data)
    graph.write_png("tree.png")
