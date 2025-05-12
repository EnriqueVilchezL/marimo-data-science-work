import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    dataset_filepath = "data/bots_vs_users.csv"
    bot_dataset = pl.read_csv(dataset_filepath, null_values=["Unknown"])
    return (bot_dataset,)


@app.cell
def _(bot_dataset, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM bot_dataset
        """
    )
    return


@app.cell
def _():
    import altair as alt
    return


if __name__ == "__main__":
    app.run()
