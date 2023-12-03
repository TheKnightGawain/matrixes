#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SizingError {
    #[error("Invalid number or size of rows: {0}")]
    Row(usize),
    #[error("Invalid number or size of columns: {0}")]
    Column(usize),
    #[error("Invalid number or size of rows ({0}) and columns ({1}).")]
    Both(usize, usize),
    #[error("Matrix is not square")]
    NotSquare,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum IndexError {
    #[error("Row value out of bounds: {0}")]
    Row(usize),
    #[error("Column value out of bounds: {0}")]
    Column(usize),
    #[error("Row and column values out of bounds: ({0}, {1})")]
    Both(usize, usize),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MinorError {
    #[error("Matrix is not square")]
    NotSquare,
    #[error("Problem with bounds: {0}")]
    BoundsError(#[from] IndexError),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum InversionError {
    #[error("Matrix was not square")]
    NotSquare,
    #[error("Matrix's determinant was 0")]
    InvalidDeterminant,
}
