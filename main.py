import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _(mo):
    mo.md("""# <center>Marimo's Tutorial</center>""")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ##1. Dataset

    <span style="color:red">LLENAR CON INTRODIUCCION AL DATASET</span>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ##2. Data loading and Transformation

    Data loading in marimo is great.


    To load data you can use any method you already. One great thing about marimo is that you have in your left tool bar access to all your data source for quick exploration. 


    For data transformation it is a bit the same, you can use everything you like from your prefered python libraries. 


    What is great is that we can take advantage of marimo reactivity to make our loading and exploration great. Let's take a look.
    """
    )
    return


@app.cell
def _(mo, pl):
    dataset_filepath = mo.notebook_location() / "data" / "bots_vs_users.csv"
    bot_dataset = pl.read_csv(
        str(dataset_filepath),
        null_values=["Unknown"],
    )
    return (bot_dataset,)


@app.cell
def _(mo):
    mo.md(r"""First of course we can simply display static data by just calling the appropiate function like schema or describe.""")
    return


@app.cell
def _(bot_dataset, mo):
    schema = bot_dataset.schema

    schema_md = "\n".join([f"- **{name}**: {dtype}" for name, dtype in schema.items()])
    mo.md(f"""
    ### DataFrame Schema
    {schema_md}""")
    return


@app.cell
def _(bot_dataset, mo):
    df_describe = bot_dataset.describe()

    mo.md(f"""
    ### Descriptive Statistics
    {df_describe.to_pandas().to_markdown()}
    """)
    return


@app.cell
def _(
    bot_dataset,
    build_column_to_display_multiselect,
    build_target_dropdown,
    mo,
):
    title_target_selector = mo.md(
        "#### But it is more fun if we make use of Marimo reactivity to explore the data wiht a filter!"
    )
    target_selector = build_target_dropdown(bot_dataset)
    selected_columns = build_column_to_display_multiselect(bot_dataset)
    mo.vstack(
        [
            title_target_selector,
            selected_columns,
            target_selector,
        ]
    )
    return selected_columns, target_selector


@app.cell
def _(display_filtered_data, selected_columns, target_selector):
    display_filtered_data(target_selector, selected_columns)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Another cool thing about marimo is that it has sql cells. From the deploy notebook you wont be able to see it, but if you run it on you own, you'll be able to play with the cell below. By default the sql query is 

    ```
    SELECT target, has_last_name 
    FROM bot_dataset 
    WHERE has_last_name = 1;
    ```
    """
    )
    return


@app.cell
def _(bot_dataset, mo):
    sql_with_marimo = mo.sql(
        f"""
            SELECT target, has_last_name
            FROM bot_dataset 
            WHERE has_last_name = 1;
        """
    )
    return (sql_with_marimo,)


@app.cell
def _(mo):
    mo.md(r"""in the sql cell you can name the output dataframe generated by you sql query, we named it `sql_with_marimo`. You can use it as a normal dataframe, so you can do all the operations you want with it, for example in the cell bellow where we are just displaying it""")
    return


@app.cell
def _(sql_with_marimo):
    sql_with_marimo
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Univariable analysis

    <span style="color:red">LLENAR CON INTRODUCCION</span>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Cleaninig the data

    <span style="color:red">LLENAR CON INTRODUCCION</span>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Multivariable analysis

    Now we are going to do some multivariable analysis with **Marimo** and **Altair**. Some common graphics are created with **Altair**, such as **heatmaps** and **bar plots** and interactivity is added with **Marimo**.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Cleaning the data

    The first thing we gotta do is clean the data. That means removing null values or replacing them with a value such as the median. In **Marimo** we can use the **Polars** or **Pandas** library to clean those values. Here, we are going to use **Polars**.
    """
    )
    return


@app.cell
def _(bot_dataset):
    bot_dataset.describe()
    return


@app.cell
def _(bot_dataset, mo, pl):
    non_useful_columns = [
        col
        for col in bot_dataset.columns
        if (bot_dataset.null_count().select(pl.col(col)).item() / len(bot_dataset)) > 0.25
    ]

    mo.md(
        rf"""
    There is a lot of **null** values in the data. However, this is true to some columns, not all of them. As a result, we decided to eliminate the columns in which the amount of nulls is equal or greater than a quarter of the amount of rows. These are: {"\n\n- "}{"\n\n- ".join(non_useful_columns)}
    """
    )
    return (non_useful_columns,)


@app.cell
def _(mo):
    mo.md(r"""Now we can filter the dataset by these columns using the `drop` method of the dataframe.""")
    return


@app.cell
def _(bot_dataset, non_useful_columns):
    multivariable_analysis_dataset = bot_dataset.drop(non_useful_columns)
    multivariable_analysis_dataset.describe()
    return (multivariable_analysis_dataset,)


@app.cell
def _(mo):
    mo.md(r"""Still, the dataset has some null values. We are going to replace them by the median of the non null values.""")
    return


@app.cell
def _(multivariable_analysis_dataset, pl):
    mv_df = multivariable_analysis_dataset.select(
        [pl.col(c).fill_null(pl.col(c).median()) for c in multivariable_analysis_dataset.columns]
    )
    return (mv_df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Correlation

    We can observe the correlation between each pair of variables, taking into account that the variables are binary. Therefore, we need to use the Phi Coefficient (\(\phi\)). The formula for the Phi coefficient (\(\phi\)) is:

    \[
    \phi = \frac{(ad - bc)}{\sqrt{(a+b)(a+c)(b+d)(c+d)}}
    \]

    Where:

    - \(a\) is the number of cases where both variables are 1.
    - \(b\) is the number of cases where the first variable is 1 and the second is 0.
    - \(c\) is the number of cases where the first variable is 0 and the second is 1.
    - \(d\) is the number of cases where both variables are 0.
    """
    )
    return


@app.cell
def _(matthews_corrcoef, mv_df, pd):
    # Convert
    pd_mv_df = mv_df.to_pandas()

    # Compute phi
    data = []
    cols = pd_mv_df.columns
    for i in cols:
        for j in cols:
            phi = matthews_corrcoef(pd_mv_df[i], pd_mv_df[j])
            data.append({"var1": i, "var2": j, "phi": phi})

    phi_df = pd.DataFrame(data)
    return (phi_df,)


@app.cell
def _(alt, mo, phi_df):
    heatmap = (
        alt.Chart(phi_df)
        .mark_rect()
        .encode(
            x=alt.X("var1:O", title=""),
            y=alt.Y("var2:O", title=""),
            color=alt.Color("phi:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
        )
        .properties(title="Phi Correlation Heatmap")
    )

    text = (
        alt.Chart(phi_df)
        .mark_text(baseline="middle")
        .encode(
            x="var1:O",
            y="var2:O",
            text=alt.Text("phi:Q", format=".2f"),
            color=alt.condition("abs(datum.phi) < 0.5", alt.value("black"), alt.value("white")),
        )
    )

    mo.ui.altair_chart((heatmap + text).configure_axis(labelFontSize=10, titleFontSize=12))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Seems that one of the most correlated variables to the target is the `has_photo` column, which makes sense since bots tend to not have profile photos.

    Also, an interesting thing to note is that some variables have a 0 even though the correlation is with themselves. This happens because the have only one value (1 or 0) in all the data samples, so the correlation with the Phi coefficient is undefined.

    These are valuable insights for a machine learning model, since those columns probably won't take any effect to predict the target.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Distribution

    We can see the distribution of data with respect to the `target`. We just need to choose one variable to see it. Fortunately, **Marimo** offers a **dropdown** box to choose from.
    """
    )
    return


@app.cell
def _(mo, mv_df):
    # UI Dropdown for feature selection
    feature_select_1 = mo.ui.dropdown(
        options=mv_df.columns,
        value="has_photo",
        label="Select Feature 1",
    )

    feature_select_2 = mo.ui.dropdown(
        options=mv_df.columns,
        value="target",
        label="Select Feature 2",
    )
    return feature_select_1, feature_select_2


@app.cell
def _(feature_select_1, feature_select_2):
    # Reactive components
    selected_feature_1 = feature_select_1.value
    selected_feature_2 = feature_select_2.value
    return selected_feature_1, selected_feature_2


@app.cell
def _(alt, mo, mv_df, selected_feature_1, selected_feature_2):
    chart_data = mv_df.to_pandas()

    mv_bar_plot_title_1 = selected_feature_1.replace("_", " ").capitalize()
    mv_bar_plot_title_2 = selected_feature_2.replace("_", " ").capitalize()

    mv_bar_plot = mo.ui.altair_chart(
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X(f"{selected_feature_2}:N", title=mv_bar_plot_title_2),
            y=alt.Y("count()", title="Count"),
            color=alt.Color(
                f"{selected_feature_1}:N",
                title=mv_bar_plot_title_1,
                legend=alt.Legend(title=mv_bar_plot_title_1),
            ),
            tooltip=["count()", f"{selected_feature_1}:N"],
        )
        .properties(title=f"Count of {selected_feature_1} by {selected_feature_2}")
    )
    return (mv_bar_plot,)


@app.cell
def _(feature_select_1, feature_select_2, mo, mv_bar_plot):
    # Display all components
    mo.vstack(
        [
            feature_select_1,
            feature_select_2,
            mv_bar_plot,
            mo.ui.table(mv_bar_plot.value),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    It is clear that some variables such as `has_photo`, `can_send_messages` or even `is_profile_closed` have a different distribution across different targets. This indicates that these variables are very suitable to determine if a user is a bot. 

    Even from this point of the analysis, possible recomendations for the social network platform would be measuring these three variables to close some accounts or give some warnings to the users.

    Finally, we also discovered some redundant variables, which are `is_profile_closed` and `access_to_closed_profile`, since they have a correlation of -1 and they are perfectly separated if the two variables are in the bar plot.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The order does not matter

    Marimo builds a DAG on your notebook, so it knows that depends on what eliminating hidden state problems. In doing so, we can put all our not important code in the end of the notebook, like an appendix. 🤗
    """
    )
    return


@app.cell
def _():
    import warnings
    return


@app.cell
def _():
    import polars as pl
    return (pl,)


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _():
    import numpy as np
    return


@app.cell
def _():
    import scipy.stats as stats
    return


@app.cell
def _():
    from sklearn.metrics import matthews_corrcoef
    return (matthews_corrcoef,)


@app.cell
def _():
    import altair as alt
    return (alt,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""For example, here I defined all the functions to build the ui for the data transformations and loading 🚀""")
    return


@app.cell
def _(mo, pl):
    def build_target_dropdown(bot_dataset: pl.dataframe) -> mo.ui.dropdown:
        unique_targets = bot_dataset["target"].unique().sort().to_list()

        options_for_target = {"Show All": "ALL"}
        for val in unique_targets:
            options_for_target[f"Target Value: {val} {'(bot)' if val == 1 else '(not a bot)'}"] = str(
                val
            )

        return mo.ui.dropdown(
            options=options_for_target,
            value="Show All",  # Default value
            label="🎯 Filter by Target Value:",
        )
    return (build_target_dropdown,)


@app.cell
def _(mo, pl):
    def build_column_to_display_multiselect(bot_dataset: pl.dataframe) -> mo.ui.multiselect:
        bot_dataset_options = {column: column for column in bot_dataset.columns}
        return mo.ui.multiselect(
            options=bot_dataset_options,
            label="Select the columns that you want to see!",
        )
    return (build_column_to_display_multiselect,)


@app.cell
def _(bot_dataset, mo, pl):
    def display_filtered_data(
        target_selector: mo.ui.dropdown,
        selected_columns: mo.ui.multiselect,
    ) -> mo.vstack:
        # handle the target selector value
        target_selector_value = target_selector.value

        if target_selector_value == "ALL":
            data_to_display = bot_dataset
            status_md = mo.md(f"### Displaying All Data ({data_to_display.height} rows)")
        else:
            filter_target_as_int = int(target_selector_value)
            data_to_display = bot_dataset.filter(pl.col("target") == filter_target_as_int)
            label = "(bots)" if filter_target_as_int == "1" else "(Not bots)"
            status_md = mo.md(
                f"""### Displaying Data for Target = **{filter_target_as_int} {label}** ({data_to_display.height} rows)"""
            )

        # handle selected columns
        columns_to_display = selected_columns.value
        if not columns_to_display:
            columns_to_display = [
                "city",
                "gender",
                "avg_likes",
                "posts_count",
                "subscribers_count",
            ]
        if "target" not in columns_to_display:
            columns_to_display.insert(0, "target")

        return mo.vstack(
            [
                status_md,
                # replace each 1 in target column by the label
                data_to_display.select(columns_to_display),
            ]
        )
    return (display_filtered_data,)


if __name__ == "__main__":
    app.run()
