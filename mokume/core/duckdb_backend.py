"""
DuckDB lazy backing for QpxDataset.

Provides the ``LazyFrame`` class — a thin wrapper around DuckDB relations
that defers computation until results are explicitly requested. This allows
QpxDataset to represent millions of PSMs/features without loading them all
into memory.

A LazyFrame can be:
- Created from a parquet file (``LazyFrame.from_parquet``)
- Created from a SQL query (``LazyFrame.from_sql``)
- Created from an existing DataFrame (``LazyFrame.from_dataframe``)
- Materialized to a pandas DataFrame on demand (``.df()``)

Key operations (filter, select, head, describe) stay lazy until ``.df()``
is called.

Example
-------
>>> lf = LazyFrame.from_parquet("data.parquet")
>>> lf.columns
['ProteinName', 'SampleID', 'Intensity', ...]
>>> lf.shape  # (row_count, col_count) — row count via COUNT(*)
(1500000, 12)
>>> filtered = lf.filter("Intensity > 0")
>>> df = filtered.df()  # Materializes to pandas DataFrame
"""

import logging
from typing import List, Optional, Union

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class LazyFrame:
    """Lazy wrapper around a DuckDB relation.

    Parameters
    ----------
    relation : duckdb.DuckDBPyRelation
        The underlying DuckDB relation.
    connection : duckdb.DuckDBPyConnection
        The DuckDB connection that owns this relation.
    source : str, optional
        Description of the data source (for repr/logging).
    """

    def __init__(
        self,
        relation: "duckdb.DuckDBPyRelation",
        connection: "duckdb.DuckDBPyConnection",
        source: str = "unknown",
        owns_connection: bool = False,
    ):
        self._relation = relation
        self._connection = connection
        self._source = source
        self._owns_connection = owns_connection
        # Cache column names (cheap to compute from relation metadata)
        self._columns: Optional[List[str]] = None
        self._row_count: Optional[int] = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(
        cls,
        path: str,
        connection: Optional["duckdb.DuckDBPyConnection"] = None,
    ) -> "LazyFrame":
        """Create a LazyFrame from a parquet file.

        Parameters
        ----------
        path : str
            Path to the parquet file.
        connection : duckdb.DuckDBPyConnection, optional
            Existing connection to reuse. If None, creates a new one.

        Returns
        -------
        LazyFrame
        """
        owns = connection is None
        if connection is None:
            connection = duckdb.connect()

        safe_path = path.replace("'", "''")
        relation = connection.sql(
            f"SELECT * FROM parquet_scan('{safe_path}')"
        )
        return cls(relation, connection, source=f"parquet:{path}", owns_connection=owns)

    @classmethod
    def from_sql(
        cls,
        sql: str,
        connection: "duckdb.DuckDBPyConnection",
        source: str = "sql",
    ) -> "LazyFrame":
        """Create a LazyFrame from an arbitrary SQL query.

        Parameters
        ----------
        sql : str
            SQL query to execute lazily.
        connection : duckdb.DuckDBPyConnection
            The DuckDB connection.
        source : str
            Description for repr.

        Returns
        -------
        LazyFrame
        """
        relation = connection.sql(sql)
        return cls(relation, connection, source=source)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        connection: Optional["duckdb.DuckDBPyConnection"] = None,
    ) -> "LazyFrame":
        """Create a LazyFrame from a pandas DataFrame.

        This registers the DataFrame as a temporary table in DuckDB,
        allowing lazy operations on in-memory data.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        connection : duckdb.DuckDBPyConnection, optional
            Existing connection. If None, creates a new one.

        Returns
        -------
        LazyFrame
        """
        owns = connection is None
        if connection is None:
            connection = duckdb.connect()

        # Use DuckDB's ability to query DataFrames directly
        relation = connection.sql("SELECT * FROM df")
        return cls(relation, connection, source="dataframe", owns_connection=owns)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def columns(self) -> List[str]:
        """Column names (lazy — extracted from relation metadata)."""
        if self._columns is None:
            self._columns = self._relation.columns
        return self._columns

    @property
    def dtypes(self) -> List[str]:
        """Column data types as strings."""
        return self._relation.dtypes

    @property
    def row_count(self) -> int:
        """Number of rows (executes COUNT(*) query)."""
        if self._row_count is None:
            result = self._relation.aggregate("COUNT(*)")
            self._row_count = result.fetchone()[0]
        return self._row_count

    @property
    def shape(self) -> tuple:
        """(rows, columns) — row count triggers a COUNT(*) query."""
        return (self.row_count, len(self.columns))

    @property
    def connection(self) -> "duckdb.DuckDBPyConnection":
        """The underlying DuckDB connection."""
        return self._connection

    @property
    def relation(self) -> "duckdb.DuckDBPyRelation":
        """The underlying DuckDB relation."""
        return self._relation

    # ------------------------------------------------------------------
    # Lazy operations
    # ------------------------------------------------------------------

    def filter(self, condition: str) -> "LazyFrame":
        """Apply a SQL WHERE filter lazily.

        Parameters
        ----------
        condition : str
            SQL condition (e.g., "Intensity > 0 AND ProteinName != 'DECOY'").

        Returns
        -------
        LazyFrame
            New LazyFrame with the filter applied.
        """
        new_relation = self._relation.filter(condition)
        return LazyFrame(new_relation, self._connection, source=f"{self._source}[filtered]")

    def select(self, *columns: str) -> "LazyFrame":
        """Select specific columns lazily.

        Parameters
        ----------
        *columns : str
            Column names to select.

        Returns
        -------
        LazyFrame
            New LazyFrame with only the selected columns.
        """
        col_str = ", ".join(f'"{c}"' for c in columns)
        new_relation = self._relation.project(col_str)
        return LazyFrame(new_relation, self._connection, source=f"{self._source}[select]")

    def head(self, n: int = 5) -> pd.DataFrame:
        """Materialize the first n rows as a DataFrame.

        Parameters
        ----------
        n : int
            Number of rows.

        Returns
        -------
        pd.DataFrame
        """
        return self._relation.limit(n).df()

    def describe(self) -> pd.DataFrame:
        """Compute basic statistics (materializes aggregation only)."""
        return self._relation.describe().df()

    def unique(self, column: str) -> List:
        """Get unique values for a column.

        Parameters
        ----------
        column : str
            Column name.

        Returns
        -------
        list
            Unique values.
        """
        result = self._relation.unique(column).df()
        return result[column].tolist()

    # ------------------------------------------------------------------
    # Materialization
    # ------------------------------------------------------------------

    def df(self) -> pd.DataFrame:
        """Materialize the full result as a pandas DataFrame.

        This triggers the actual computation.

        Returns
        -------
        pd.DataFrame
        """
        logger.debug("Materializing LazyFrame (%s)", self._source)
        result = self._relation.df()
        self._row_count = len(result)
        return result

    def to_arrow(self):
        """Materialize as a PyArrow Table."""
        return self._relation.arrow()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying DuckDB connection if owned by this LazyFrame."""
        if self._owns_connection and self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        cols = self.columns
        ncols = len(cols)
        col_preview = ", ".join(cols[:5])
        if ncols > 5:
            col_preview += f", ... ({ncols - 5} more)"
        return f"LazyFrame(source={self._source}, cols=[{col_preview}])"

    def __len__(self) -> int:
        return self.row_count

    def __contains__(self, item: str) -> bool:
        """Check if a column name exists."""
        return item in self.columns


def is_lazy(obj) -> bool:
    """Check if an object is a LazyFrame.

    Parameters
    ----------
    obj : any
        Object to check.

    Returns
    -------
    bool
    """
    return isinstance(obj, LazyFrame)


def ensure_dataframe(obj: Union[pd.DataFrame, "LazyFrame"]) -> pd.DataFrame:
    """Convert a LazyFrame to DataFrame if needed.

    Parameters
    ----------
    obj : pd.DataFrame or LazyFrame
        Input data.

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(obj, LazyFrame):
        return obj.df()
    return obj
