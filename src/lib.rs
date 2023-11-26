mod errors;

extern crate num;

pub use errors::{BoundsError as BE, InversionError as IE, MatrixError as ME, SizingError as SE};
use num::{One, Zero};
use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, Index, IndexMut, Mul, Neg, Sub, SubAssign},
};

#[derive(Clone, PartialEq, Eq)]
pub struct Matrix<T>
where
    T: Copy,
{
    data: Vec<T>,
    rows: usize,
    columns: usize,
}

// constructors
impl<T> Matrix<T>
where
    T: Copy,
{
    /// Returns a new matrix with the specified rows and columns initialized to 0 or a sizing error.
    ///
    /// # Errors
    ///
    /// Niether rows nor columns can be 0.
    pub fn new(rows: usize, columns: usize) -> Result<Self, SE>
    where
        T: Zero,
    {
        if rows == 0 && columns == 0 {
            return Err(SE::Both(rows, columns));
        }

        if rows == 0 {
            return Err(SE::Row(0));
        }

        if columns == 0 {
            return Err(SE::Column(0));
        }

        Ok(Self {
            data: vec![T::zero(); rows * columns],
            rows: rows,
            columns: columns,
        })
    }

    /// Returns an Option to a new identity matrix with dimensions n x n.
    ///
    /// # Errors
    ///
    /// n cannot be 0.
    pub fn new_identity(n: usize) -> Option<Self>
    where
        T: Zero + One,
    {
        if n == 0 {
            return None;
        }

        Some(Self {
            data: (0..n.pow(2))
                .map(|i| {
                    if i % (n + 1) == 0 {
                        T::one()
                    } else {
                        T::zero()
                    }
                })
                .collect(),
            rows: n,
            columns: n,
        })
    }

    /// Creates a matrix with the specified columns from data, which must be in row-major order.
    ///
    /// # Errors
    ///
    /// columns cannot be zero. data must have non-zero length that is divisable by columns.
    pub fn new_with_data(columns: usize, data: Vec<T>) -> Result<Self, SE> {
        let len = data.len();

        if columns == 0 && len == 0 {
            return Err(SE::Both(0, 0));
        }

        if columns == 0 {
            return Err(SE::Column(0));
        }

        if len == 0 || len % columns != 0 {
            return Err(SE::Row(len % columns));
        }

        Ok(Self {
            data,
            rows: len / columns,
            columns: columns,
        })
    }

    /// Creates a matrix from data, which is a vec of rows of elements.
    ///
    /// # Errors
    ///
    /// data must have elements and elements must the same length
    pub fn new_from_data(data: &[Vec<T>]) -> Result<Self, SE> {
        let rows = data.len();

        if rows == 0 {
            return Err(SE::Both(0, 0));
        }

        let columns = data[0].len();

        if columns == 0 {
            return Err(SE::Row(0));
        }

        let mut elements: Vec<T> = Vec::new();

        for r in data {
            if r.len() != columns {
                return Err(SE::Row(r.len()));
            }

            for e in r {
                elements.push(*e);
            }
        }

        Ok(Self {
            data: elements,
            rows,
            columns,
        })
    }
}

// getters
impl<T> Matrix<T>
where
    T: Copy,
{
    /// Returns the data as a vec of references.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns the row count.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the column count.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the number of items in the matrix.
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the matrix has the same number of rows and columns.
    pub fn is_square(&self) -> bool {
        self.rows == self.columns
    }

    /// Returns indexed row as a vec of references.
    ///
    /// # Errors
    ///
    /// index cannot reference a row that does not exist.
    pub fn get_row(&self, index: usize) -> Option<Vec<&T>> {
        if index >= self.rows {
            return None;
        }

        Some((0..self.columns).map(|c| &self[(index, c)]).collect())
    }

    /// Returns a vec of rows indexed by the range.
    ///
    /// # Error
    ///
    /// range cannot refer to a row that does not exist.
    pub fn get_rows(&self, range: impl Iterator<Item = usize>) -> Option<Vec<Vec<&T>>> {
        range.map(|r| self.get_row(r)).collect::<Option<Vec<_>>>()
    }

    /// Returns indexed column as a vec of references.
    ///
    /// # Error
    ///
    /// index cannot refer to a column that does not exist.
    pub fn get_column(&self, index: usize) -> Option<Vec<&T>> {
        if index >= self.columns {
            return None;
        }

        Some((0..self.rows).map(|r| &self[(r, index)]).collect())
    }

    /// Returns a vec of columns indexed by the range.
    ///
    /// # Errors
    ///
    /// range cannot refer to a column that does not exist.
    pub fn get_columns(&self, range: impl Iterator<Item = usize>) -> Option<Vec<Vec<&T>>> {
        range
            .map(|c| self.get_column(c))
            .collect::<Option<Vec<Vec<&T>>>>()
    }
}

// mut getters
impl<T> Matrix<T>
where
    T: Copy,
{
    /// Returns data as a vec of mutable references.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns indexed row as a vec of mutable references.
    ///
    /// # Errors
    ///
    /// index cannot refer to a row that does not exist.
    pub fn get_mut_row(&mut self, index: usize) -> Option<Vec<&mut T>> {
        if index >= self.rows {
            return None;
        }

        Some(
            self.data[(index * self.columns)..((index + 1) * self.columns)]
                .iter_mut()
                .collect(),
        )
    }

    /// Returns indexed column as a vec of matable references.
    ///
    /// # Errors
    ///
    /// index cannot refer to a column that does not exist.
    pub fn get_mut_column(&mut self, index: usize) -> Option<Vec<&mut T>> {
        if index >= self.columns {
            return None;
        }

        Some(
            self.data
                .iter_mut()
                .skip(index)
                .step_by(self.columns)
                .collect(),
        )
    }
}

// swappers
impl<T> Matrix<T>
where
    T: Copy + 'static,
{
    /// Swaps the indexed elements.
    ///
    /// # Errors
    ///
    /// el1 and el2 must refer to valid elements of the matrix
    pub fn swap_elements(&mut self, el1: (usize, usize), el2: (usize, usize)) -> Option<BE> {
        if el1.0 >= self.rows && el1.1 >= self.columns {
            return Some(BE::Both(el1.0, el1.1));
        }

        if el1.0 >= self.rows {
            return Some(BE::Row(el1.0));
        }

        if el1.1 >= self.columns {
            return Some(BE::Column(el1.1));
        }

        if el2.0 >= self.rows && el2.1 >= self.columns {
            return Some(BE::Both(el2.0, el2.1));
        }

        if el2.0 >= self.rows {
            return Some(BE::Row(el2.0));
        }

        if el2.1 >= self.columns {
            return Some(BE::Column(el2.1));
        }

        let temp = self[el1];
        self[el1] = self[el2];
        self[el2] = temp;

        None
    }

    /// Swaps the indexed rows.
    ///
    /// # Errors
    ///
    /// row1 and row2 must refer to rows that exist
    pub fn swap_rows(&mut self, row1: usize, row2: usize) -> Option<BE> {
        let first = match self.get_row(row1) {
            Some(t) => t,
            None => return Some(BE::Row(row1)),
        }
        .iter()
        .map(|&e| *e)
        .collect::<Vec<_>>();

        let second = match self.get_row(row2) {
            Some(t) => t,
            None => return Some(BE::Row(row2)),
        }
        .iter()
        .map(|&e| *e)
        .collect::<Vec<_>>();

        self.get_mut_row(row1)
            .unwrap()
            .iter_mut()
            .zip(second.iter())
            .for_each(|(f, s)| **f = *s);

        self.get_mut_row(row2)
            .unwrap()
            .iter_mut()
            .zip(first.iter())
            .for_each(|(s, f)| **s = *f);

        None
    }

    /// Swaps the indexed columns.
    ///
    /// # Errors
    ///
    /// col1 and col2 must refer to columns that exist.
    pub fn swap_columns(&mut self, col1: usize, col2: usize) -> Option<BE> {
        let first = match self.get_column(col1) {
            Some(t) => t,
            None => return Some(BE::Column(col1)),
        }
        .iter()
        .map(|&e| *e)
        .collect::<Vec<_>>();
        let second = match self.get_column(col2) {
            Some(t) => t,
            None => return Some(BE::Column(col2)),
        }
        .iter()
        .map(|&e| *e)
        .collect::<Vec<_>>();

        self.get_mut_column(col1)
            .unwrap()
            .iter_mut()
            .zip(second.iter())
            .for_each(|(f, s)| **f = *s);

        self.get_mut_column(col2)
            .unwrap()
            .iter_mut()
            .zip(first.iter())
            .for_each(|(s, f)| **s = *f);

        None
    }
}

// operations
impl<T> Matrix<T>
where
    T: Copy + Mul<Output = T>,
{
    /// Multiplies each element of the matrix by factor.
    pub fn scale(&mut self, factor: T) {
        self.data.iter_mut().for_each(|e| *e = *e * factor);
    }

    /// Multiplies each element of indexed row by factor.
    ///
    /// # Errors
    ///
    /// row must refer to a row that exists.
    ///
    /// # Panics
    ///
    /// Multiplication by factor may panic.
    pub fn scale_row(&mut self, row: usize, factor: T) -> Option<BE> {
        match self.get_mut_row(row) {
            Some(t) => t,
            None => return Some(BE::Row(row)),
        }
        .iter_mut()
        .for_each(|t| **t = **t * factor);

        None
    }

    /// Adds source row scaled by factor to target row.
    ///
    /// # Errors
    ///
    /// source and target must refer to rows that exist.
    ///
    /// # Panics
    ///
    /// Multiplication by factor may panic.
    pub fn add_scaled_row(&mut self, source: usize, target: usize, factor: T) -> Option<BE>
    where
        T: Add<Output = T>,
    {
        let source_row = match self.get_row(source) {
            Some(t) => t,
            None => return Some(BE::Row(source)),
        }
        .iter()
        .map(|&e| *e)
        .collect::<Vec<_>>();

        match self.get_mut_row(target) {
            Some(t) => t,
            None => return Some(BE::Row(target)),
        }
        .iter_mut()
        .zip(source_row.iter())
        .for_each(|(t, s)| **t = **t + (*s * factor));

        None
    }

    /// Multiplies each element of indexed row by factor.
    ///
    /// # Errors
    ///
    /// column must refer to a row that exists.
    ///
    /// # Panics
    ///
    /// Multiplication by factor may panic.
    pub fn scale_column(&mut self, column: usize, factor: T) -> Option<BE> {
        match self.get_mut_column(column) {
            Some(t) => t,
            None => return Some(BE::Column(column)),
        }
        .iter_mut()
        .for_each(|t| **t = **t * factor);

        None
    }

    /// Adds source row scaled by factor to target row.
    ///
    /// # Errors
    ///
    /// source and target must refer to rows that exist.
    ///
    /// # Panics
    ///
    /// Multiplication by factor may panic.
    pub fn add_scaled_column(&mut self, source: usize, target: usize, factor: T) -> Option<BE>
    where
        T: Add<Output = T>,
    {
        let source_row = match self.get_column(source) {
            Some(t) => t,
            None => return Some(BE::Column(source)),
        }
        .iter()
        .map(|&e| *e)
        .collect::<Vec<_>>();

        match self.get_mut_column(target) {
            Some(t) => t,
            None => return Some(BE::Column(target)),
        }
        .iter_mut()
        .zip(source_row.iter())
        .for_each(|(t, s)| **t = **t + (*s * factor));

        None
    }

    pub fn resize(&mut self, bounds: (usize, usize)) -> Option<BE> {
        if bounds.0 == 0 {
            return Some(BE::Row(0));
        }

        if bounds.1 == 0 {
            return Some(BE::Column(0));
        }

        if bounds.0 * bounds.1 != self.size() {
            return Some(BE::Both(bounds.0, bounds.1));
        }

        self.rows = bounds.0;
        self.columns = bounds.1;
        None
    }

    /// Removes the data of the selected row and changes to bounds to match
    ///
    /// Errors
    ///
    /// row must refer to a row that exists
    pub fn remove_row(&mut self, row: usize) -> Option<usize> {
        if row >= self.rows {
            return Some(row);
        }

        self.data
            .drain((row * self.columns)..((row + 1) * self.columns));
        self.rows -= 1;

        None
    }

    /// Removes the data of the selected column and changes to bounds to match
    ///
    /// Errors
    ///
    /// column must refer to a row that exists
    pub fn remove_column(&mut self, column: usize) -> Option<usize> {
        if column >= self.columns {
            return Some(column);
        }

        for r in 0..self.rows {
            self.data.remove(r * (self.columns - 1) + column);
        }

        self.columns -= 1;

        None
    }
}

// derivers
impl<T> Matrix<T>
where
    T: Copy,
{
    /// Creates a transpose matrix, whose rows are equivalent to the base matrix's columns.
    pub fn transpose(&self) -> Self {
        let mut elements: Vec<T> = vec![];

        for c in self.get_columns(0..self.columns).unwrap() {
            elements.append(&mut c.iter().map(|e| **e).collect());
        }

        Matrix {
            data: elements,
            rows: self.columns,
            columns: self.rows,
        }
    }

    /// Returns the determinant of the matrix made from all elements that are not in the same column or row as the indexed element.
    ///
    /// # Errors
    ///
    /// matrix must be square and indexed element must exist.
    pub fn minor(&self, element: (usize, usize)) -> Result<T, ME>
    where
        T: Mul<Output = T> + Sub<Output = T> + Zero,
    {
        if !self.is_square() {
            return Err(SE::NotSquare.into());
        }

        if element.0 >= self.rows && element.1 >= self.columns {
            return Err(BE::Both(element.0, element.1).into());
        }

        if element.0 >= self.rows {
            return Err(BE::Row(element.0).into());
        }

        if element.1 >= self.columns {
            return Err(BE::Column(element.1).into());
        }

        let mut copy = self.clone();

        copy.remove_row(element.0);
        copy.remove_column(element.1);

        Ok(copy.determinant().unwrap())
    }

    /// Returns a new matrix constructed of the minors of each element of the base matrix.
    ///
    /// # Errors
    ///
    /// Matrix must be square.
    pub fn minor_matrix(&self) -> Option<Self>
    where
        T: Mul<Output = T> + Sub<Output = T> + Zero,
    {
        if !self.is_square() {
            return None;
        }

        Matrix::new_with_data(
            self.columns,
            (0..(self.rows * self.columns))
                .map(|n| {
                    self.minor(((n / self.columns), (n % self.columns)))
                        .unwrap()
                })
                .collect(),
        )
        .ok()
    }

    /// Returns the minor matrix with every other element negated.
    ///
    /// Matrix must be square.
    pub fn cofactor(&self) -> Option<Self>
    where
        T: Neg<Output = T> + Mul<Output = T> + Sub<Output = T> + Zero,
    {
        let mut out = self.minor_matrix()?;

        out.data.iter_mut().enumerate().for_each(|(n, e)| {
            if (n / self.columns + n % self.columns) % 2 == 1 {
                *e = e.neg();
            }
        });

        Some(out)
    }

    /// Returns the transposeof the cofactor of the matrix.
    ///
    /// # Errors
    ///
    /// Matrix must be square.
    pub fn adjunct(&self) -> Option<Self>
    where
        T: Neg<Output = T> + Mul<Output = T> + Sub<Output = T> + Zero,
    {
        Some(self.cofactor()?.transpose())
    }

    /// Returns the determinant of the matrix.
    ///
    /// # Errors
    ///
    /// Matrix must be square.
    pub fn determinant(&self) -> Option<T>
    where
        T: Mul<Output = T> + Zero + Sub<Output = T>,
    {
        if !self.is_square() {
            return None;
        }

        if self.rows == 1 {
            return Some(self.data[0]);
        }

        Some(
            self.get_row(0)
                .unwrap()
                .iter()
                .enumerate()
                .fold(T::zero(), |res, (c, &e)| {
                    let det = *e * self.minor((0, c)).unwrap();
                    if c % 2 == 0 {
                        res + det
                    } else {
                        res - det
                    }
                }),
        )
    }

    /// Returns the adjunct scaled by the inverse of the determinant.
    ///
    /// # Errors
    ///
    /// Matrix must be square, determinant must not be zero
    pub fn inverse(&self) -> Result<Self, IE>
    where
        T: Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + Neg<Output = T>
            + Zero
            + One
            + PartialEq,
    {
        let det = self.determinant().ok_or(IE::NotSquare)?;

        if det == T::zero() {
            return Err(IE::InvalidDeterminant);
        }

        let mut out = self.adjunct().unwrap();
        out.scale(T::one() / det);

        Ok(out)
    }

    /// Returns the adjunct scaled by the inverse of the derminant.
    ///
    /// # Errors
    ///
    /// Matrix must be square, determinant must not be zero
    ///
    /// # Warning
    ///
    /// May not work on integers, or other types where a / b * b does not always equal a
    pub fn fast_inverse(&self) -> Result<Self, IE>
    where
        T: Copy + Zero + One + Div<Output = T> + Neg<Output = T> + PartialEq,
    {
        if !self.is_square() {
            return Err(IE::NotSquare);
        }

        let mut clone = self.clone();
        let mut out = Matrix::new_identity(clone.rows).unwrap();

        for c in 0..clone.rows {
            if !T::is_one(&clone[(c, c)]) {
                if T::is_zero(&clone[(c, c)]) {
                    return Err(IE::InvalidDeterminant);
                }
                let factor = T::one() / clone[(c, c)];
                clone.scale_row(c, factor);
                out.scale_row(c, factor);
            }

            for r in 0..c {
                if !T::is_zero(&clone[(r, c)]) {
                    let factor = clone[(r, c)].neg();
                    clone.add_scaled_row(c, r, factor);
                    out.add_scaled_row(c, r, factor);
                }
            }

            for r in (c + 1)..clone.rows {
                if !T::is_zero(&clone[(r, c)]) {
                    let factor = clone[(r, c)].neg();
                    clone.add_scaled_row(c, r, factor);
                    out.add_scaled_row(c, r, factor);
                }
            }
        }

        Ok(out)
    }

    /// Returns a new matrix with the same data, but rows and columns of bounds or the amount of data such bounds would fit.
    ///
    /// Errors
    ///
    /// Matrix of size bounds must fit the same amount of data as this.
    pub fn as_resize(&self, bounds: (usize, usize)) -> Result<Matrix<T>, BE> {
        if bounds.0 == 0 {
            return Err(BE::Row(0));
        }

        if bounds.1 == 0 {
            return Err(BE::Column(0));
        }

        if bounds.0 * bounds.1 != self.size() {
            return Err(BE::Both(bounds.0, bounds.1));
        }

        Ok(Matrix {
            data: self.data.clone(),
            rows: bounds.0,
            columns: bounds.1,
        })
    }
}

impl<T> Debug for Matrix<T>
where
    T: Copy + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.rows == 1 {
            return write!(f, "{:?}", self.data);
        }

        writeln!(f, "{:?}", self.get_row(0).unwrap())?;
        let middle = self.get_rows(1..(self.rows - 1)).unwrap();
        for row in middle {
            writeln!(f, "{:?}", row)?;
        }
        write!(f, "{:?}", self.get_row(self.rows - 1).unwrap())
    }
}

impl<T> Index<(usize, usize)> for Matrix<T>
where
    T: Copy,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(
            index.0 < self.rows && index.1 < self.columns,
            "Index out of bounds. index: {:?}, matrix bounds: {:?}",
            index,
            (self.rows, self.columns)
        );

        &self.data[index.0 * self.columns + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
where
    T: Copy,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(
            index.0 < self.rows && index.1 < self.columns,
            "Index out of bounds. index: {:?}, matrix bounds: {:?}",
            index,
            (self.rows, self.columns)
        );

        &mut self.data[index.0 * self.columns + index.1]
    }
}

impl<T> AddAssign for Matrix<T>
where
    T: Add<Output = T> + Copy,
{
    /// # Panics
    ///
    /// rhs must have the same rows and columns as the matrix.
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.rows, rhs.rows, "Mismatched rows.");
        assert_eq!(self.columns, rhs.columns, "Mismatched columns");

        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(s, o)| *s = *s + *o);
    }
}

impl<T, U> Add for Matrix<T>
where
    T: Add<Output = U> + Copy,
    U: Copy,
{
    type Output = Result<Matrix<U>, SE>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.rows != rhs.rows && self.columns != rhs.columns {
            return Err(SE::Both(rhs.rows, rhs.columns));
        }

        if self.rows != rhs.rows {
            return Err(SE::Row(rhs.rows));
        }

        if self.columns != rhs.columns {
            return Err(SE::Column(rhs.columns));
        }

        Ok(Matrix {
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(s, r)| *s + *r)
                .collect(),
            rows: self.rows,
            columns: self.columns,
        })
    }
}

impl<T, U> Add for &Matrix<T>
where
    T: Add<Output = U> + Copy,
    U: Copy,
{
    type Output = Result<Matrix<U>, SE>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.rows != rhs.rows && self.columns != rhs.columns {
            return Err(SE::Both(rhs.rows, rhs.columns));
        }

        if self.rows != rhs.rows {
            return Err(SE::Row(rhs.rows));
        }

        if self.columns != rhs.columns {
            return Err(SE::Column(rhs.columns));
        }

        Ok(Matrix {
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(s, r)| *s + *r)
                .collect(),
            rows: self.rows,
            columns: self.columns,
        })
    }
}

impl<T> SubAssign for Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    fn sub_assign(&mut self, rhs: Self) {
        assert!(self.rows == rhs.rows, "Mismatched rows");
        assert!(self.columns == rhs.columns, "Mismatched columns");

        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(s, o)| *s = *s - *o);
    }
}

impl<T, U> Sub for Matrix<T>
where
    T: Sub<Output = U> + Copy,
    U: Copy,
{
    type Output = Result<Matrix<U>, SE>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.rows != rhs.rows && self.columns != rhs.columns {
            return Err(SE::Both(rhs.rows, rhs.columns));
        }

        if self.rows != rhs.rows {
            return Err(SE::Row(rhs.rows));
        }

        if self.columns != rhs.columns {
            return Err(SE::Column(rhs.columns));
        }

        Ok(Matrix {
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(s, r)| *s - *r)
                .collect(),
            rows: self.rows,
            columns: self.columns,
        })
    }
}

impl<T, U> Sub for &Matrix<T>
where
    T: Sub<Output = U> + Copy,
    U: Copy,
{
    type Output = Result<Matrix<U>, SE>;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.rows != rhs.rows && self.columns != rhs.columns {
            return Err(SE::Both(rhs.rows, rhs.columns));
        }

        if self.rows != rhs.rows {
            return Err(SE::Row(rhs.rows));
        }

        if self.columns != rhs.columns {
            return Err(SE::Column(rhs.columns));
        }

        Ok(Matrix {
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(s, r)| *s - *r)
                .collect(),
            rows: self.rows,
            columns: self.columns,
        })
    }
}

impl<T, U> Mul for Matrix<T>
where
    T: Copy + Mul<Output = U>,
    U: Copy + Zero,
{
    type Output = Result<Matrix<U>, SE>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.columns != rhs.rows {
            return Err(SE::Row(rhs.rows));
        }

        Matrix::new_from_data(
            &(0..self.rows)
                .map(|r| {
                    (0..rhs.columns)
                        .map(|c| {
                            self.get_row(r)
                                .unwrap()
                                .iter()
                                .zip(rhs.get_column(c).unwrap().iter())
                                .fold(U::zero(), |res, (&el, &er)| res + (*el * *er))
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )
    }
}

impl<T, U> Mul for &Matrix<T>
where
    T: Copy + Mul<Output = U>,
    U: Copy + Zero,
{
    type Output = Result<Matrix<U>, SE>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.columns != rhs.rows {
            return Err(SE::Row(rhs.rows));
        }

        Matrix::new_from_data(
            &(0..self.rows)
                .map(|r| {
                    (0..rhs.columns)
                        .map(|c| {
                            self.get_row(r)
                                .unwrap()
                                .iter()
                                .zip(rhs.get_column(c).unwrap().iter())
                                .fold(U::zero(), |res, (&el, &er)| res + (*el * *er))
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )
    }
}

impl<T, U> Neg for Matrix<T>
where
    T: Neg<Output = U> + Copy,
    U: Copy,
{
    type Output = Matrix<U>;

    fn neg(self) -> Self::Output {
        Matrix {
            data: self.data.iter().map(|t| t.neg()).collect(),
            rows: self.rows,
            columns: self.columns,
        }
    }
}

impl<T, U> Neg for &Matrix<T>
where
    T: Neg<Output = U> + Copy,
    U: Copy,
{
    type Output = Matrix<U>;

    fn neg(self) -> Self::Output {
        Matrix {
            data: self.data.iter().map(|t| t.neg()).collect(),
            rows: self.rows,
            columns: self.columns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod constructors {
        use super::*;

        mod new {
            use super::*;

            #[test]
            fn handles_errors() {
                assert_eq!(Matrix::<u8>::new(0, 5), Err(SE::Row(0)));
                assert_eq!(Matrix::<i32>::new(2, 0), Err(SE::Column(0)));
                assert_eq!(Matrix::<usize>::new(0, 0), Err(SE::Both(0, 0)));
            }

            #[test]
            fn creates_matrix() {
                assert_eq!(
                    Matrix::<i8>::new(2, 5),
                    Ok(Matrix {
                        data: vec![0; 10],
                        rows: 2,
                        columns: 5
                    })
                );
                assert_eq!(
                    Matrix::new(8, 3),
                    Ok(Matrix {
                        data: vec![0; 24],
                        rows: 8,
                        columns: 3
                    })
                );
                assert_eq!(
                    Matrix::new(4, 4),
                    Ok(Matrix {
                        data: vec![0; 16],
                        rows: 4,
                        columns: 4
                    })
                );
            }
        }

        mod new_identity {
            use super::*;

            #[test]
            fn handles_errors() {
                assert_eq!(Matrix::<i128>::new_identity(0), None);
            }

            #[test]
            fn creates_matrix() {
                assert_eq!(
                    Matrix::new_identity(4),
                    Some(Matrix {
                        data: vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                        rows: 4,
                        columns: 4
                    })
                );
            }
        }

        mod new_with_data {
            use super::*;

            #[test]
            fn handles_errors() {
                assert_eq!(
                    Matrix::new_with_data(0, vec![1, 2, 3, 4]),
                    Err(SE::Column(0))
                );
                assert_eq!(Matrix::<u8>::new_with_data(4, vec![]), Err(SE::Row(0)));
                assert_eq!(Matrix::new_with_data(3, vec![5; 7]), Err(SE::Row(1)));
                assert_eq!(Matrix::<i16>::new_with_data(0, vec![]), Err(SE::Both(0, 0)))
            }

            #[test]
            fn creates_matrix() {
                assert_eq!(
                    Matrix::new_with_data(7, (0u32..35).collect()),
                    Ok(Matrix {
                        data: (0u32..35).collect(),
                        rows: 5,
                        columns: 7
                    })
                )
            }
        }

        mod new_from_data {
            use super::*;

            #[test]
            fn handles_errors() {
                assert_eq!(Matrix::<&str>::new_from_data(&vec![]), Err(SE::Both(0, 0)));
                assert_eq!(
                    Matrix::new_from_data(&vec![vec![], vec![1, 5, 6], vec![2, 6, 9]]),
                    Err(SE::Row(0))
                );
                assert_eq!(
                    Matrix::new_from_data(&vec![
                        vec![1, 5, 3, 2, 7],
                        vec![1, 2, 45, 7, 3],
                        vec![65, 8, 5, 23, 67],
                        vec![123, 5, 47]
                    ]),
                    Err(SE::Row(3))
                )
            }

            #[test]
            fn creates_matrix() {
                assert_eq!(
                    Matrix::new_from_data(&vec![
                        vec![1, 2, 3, 4],
                        vec![2, 3, 4, 1],
                        vec![3, 4, 1, 2],
                        vec![4, 1, 2, 3]
                    ]),
                    Ok(Matrix {
                        data: vec![1, 2, 3, 4, 2, 3, 4, 1, 3, 4, 1, 2, 4, 1, 2, 3],
                        rows: 4,
                        columns: 4
                    })
                );
                assert_eq!(
                    Matrix::new_from_data(&vec![vec![4, 2, 1, 5, 3], vec![1, 2, 3, 4, 5],]),
                    Ok(Matrix {
                        data: vec![4, 2, 1, 5, 3, 1, 2, 3, 4, 5],
                        rows: 2,
                        columns: 5
                    })
                );
            }
        }
    }

    mod getters {
        use super::*;

        #[test]
        fn data() {
            let m = Matrix::<u16>::new(12, 5).unwrap();

            assert_eq!(m.data(), vec![0; 60]);
        }

        #[test]
        fn rows() {
            let m = Matrix::<u16>::new(9, 3).unwrap();

            assert_eq!(m.rows(), 9);
        }

        #[test]
        fn columns() {
            let m = Matrix::<u16>::new(2, 4).unwrap();

            assert_eq!(m.columns(), 4);
        }

        #[test]
        fn size() {
            let m1 = Matrix::<i32>::new(9, 16).unwrap();
            let m2 = Matrix::<u32>::new_identity(12).unwrap();

            assert_eq!(m1.size(), 144);
            assert_eq!(m2.size(), 144);
        }

        #[test]
        fn is_square() {
            let m1 = Matrix::<u16>::new(6, 2).unwrap();
            let m2 = Matrix::<i8>::new(3, 3).unwrap();
            let m3 = Matrix::<u32>::new_identity(4).unwrap();

            assert!(!m1.is_square());
            assert!(m2.is_square());
            assert!(m3.is_square());
        }

        mod get_row {
            use super::*;

            #[test]
            fn handles_errors() {
                let m = Matrix::new_from_data(&vec![vec![0]]).unwrap();

                assert_eq!(m.get_row(3), None);
            }

            #[test]
            fn gets_row() {
                let m = Matrix::<i32>::new_identity(5).unwrap();

                assert_eq!(m.get_row(3).unwrap(), vec![&0i32, &0, &0, &1, &0]);
            }
        }

        mod get_rows {
            use super::*;

            #[test]
            fn handles_errors() {
                let m = Matrix::<u8>::new_identity(4).unwrap();

                assert_eq!(m.get_rows(0..8), None);
            }

            #[test]
            fn gets_rows() {
                let m = Matrix::<u64>::new_identity(7).unwrap();

                assert_eq!(
                    m.get_rows(1..4).unwrap(),
                    vec![
                        vec![&0, &1, &0, &0, &0, &0, &0],
                        vec![&0, &0, &1, &0, &0, &0, &0],
                        vec![&0, &0, &0, &1, &0, &0, &0]
                    ]
                )
            }
        }

        mod get_column {
            use super::*;

            #[test]
            fn handles_errors() {
                let m = Matrix::new_from_data(&vec![
                    vec![1, 2, 3, 4, 5],
                    vec![6, 7, 8, 7, 6],
                    vec![5, 4, 3, 2, 1],
                ])
                .unwrap();

                assert_eq!(m.get_column(5), None);
            }

            #[test]
            fn gets_column() {
                let m = Matrix::new_from_data(&vec![
                    vec![1, 2, 3, 4, 5],
                    vec![6, 7, 8, 7, 6],
                    vec![5, 4, 3, 2, 1],
                ])
                .unwrap();

                assert_eq!(m.get_column(3).unwrap(), vec![&4, &7, &2]);
            }
        }

        mod get_columns {
            use super::*;

            #[test]
            fn handles_errors() {
                let m = Matrix::new_from_data(&vec![
                    vec![1, 2, 3, 4, 5],
                    vec![6, 7, 8, 7, 6],
                    vec![5, 4, 3, 2, 1],
                ])
                .unwrap();

                assert_eq!(m.get_columns(0..9), None);
            }

            #[test]
            fn gets_columns() {
                let m = Matrix::new_from_data(&vec![
                    vec![1, 2, 3, 4, 5],
                    vec![6, 7, 8, 7, 6],
                    vec![5, 4, 3, 2, 1],
                ])
                .unwrap();

                assert_eq!(
                    m.get_columns(2..4).unwrap(),
                    vec![vec![&3, &8, &3], vec![&4, &7, &2]]
                );
            }
        }
    }

    mod mut_getters {
        use super::*;

        #[test]
        fn data_mut() {
            let mut m = Matrix::new_identity(2).unwrap();
            let data = m.data_mut();

            assert_eq!(data, &mut vec![1, 0, 0, 1]);

            data[1] = 5;

            assert_eq!(m.data, vec![1, 5, 0, 1]);
        }

        mod get_mut_row {
            use super::*;

            #[test]
            fn handles_errors() {
                let mut m = Matrix::<i128>::new(7, 5).unwrap();

                assert_eq!(m.get_mut_row(23), None);
            }

            #[test]
            fn gets_mut_row() {
                let mut m = Matrix::<i8>::new_identity(3).unwrap();
                let mut row = m.get_mut_row(2).unwrap();

                assert_eq!(row, vec![&mut 0, &mut 0, &mut 1]);

                *row[0] = 3;

                assert_eq!(m.get_row(2).unwrap(), vec![&3, &0, &1]);
            }
        }

        mod get_mut_column {
            use super::*;

            #[test]
            fn handles_errors() {
                let mut m = Matrix::<u8>::new(5, 8).unwrap();

                assert_eq!(m.get_mut_column(19), None);
            }

            #[test]
            fn gets_column() {
                let mut m = Matrix::new_from_data(&vec![
                    vec![1, 2, 3, 4, 5, 6, 7],
                    vec![8, 9, 10, 11, 12, 13, 14],
                    vec![14, 13, 12, 11, 10, 9, 8],
                    vec![7, 6, 5, 4, 3, 2, 1],
                ])
                .unwrap();
                let mut col = m.get_mut_column(5).unwrap();

                assert_eq!(col, vec![&mut 6, &mut 13, &mut 9, &mut 2]);

                *col[3] = 17;

                assert_eq!(m.get_column(5).unwrap(), vec![&6, &13, &9, &17]);
            }
        }
    }

    mod swappers {
        use super::*;

        mod swap_elements {
            use super::*;

            #[test]
            fn handles_errors() {
                let mut m = Matrix::<u16>::new_identity(6).unwrap();

                assert_eq!(m.swap_elements((13, 2), (5, 1)), Some(BE::Row(13)));
                assert_eq!(m.swap_elements((0, 8), (3, 4)), Some(BE::Column(8)));
                assert_eq!(m.swap_elements((18, 27), (1, 2)), Some(BE::Both(18, 27)));
                assert_eq!(m.swap_elements((3, 0), (6, 5)), Some(BE::Row(6)));
                assert_eq!(m.swap_elements((4, 3), (3, 9)), Some(BE::Column(9)));
                assert_eq!(m.swap_elements((0, 2), (12, 7)), Some(BE::Both(12, 7)));
            }

            #[test]
            fn swaps_elements() {
                let mut m = Matrix::<i8>::new_identity(3).unwrap();

                assert_eq!(m.swap_elements((0, 0), (0, 2)), None);
                assert_eq!(m.data, vec![0, 0, 1, 0, 1, 0, 0, 0, 1]);
            }
        }

        mod swap_rows {
            use super::*;

            #[test]
            fn handles_errors() {
                let mut m = Matrix::<i64>::new_identity(7).unwrap();

                assert_eq!(m.swap_rows(2, 9), Some(BE::Row(9)));
                assert_eq!(m.swap_rows(7, 4), Some(BE::Row(7)));
            }

            #[test]
            fn swaps_rows() {
                let mut m = Matrix::new_from_data(&vec![
                    vec![5, 4, 3, 2, 1],
                    vec![1, 2, 3, 4, 5],
                    vec![5, 4, 3, 2, 1],
                ])
                .unwrap();

                assert_eq!(m.swap_rows(0, 1), None);
                assert_eq!(m.data, vec![1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1]);
            }
        }

        mod swap_columns {
            use super::*;

            #[test]
            fn handles_errors() {
                let mut m = Matrix::<u16>::new(7, 3).unwrap();

                assert_eq!(m.swap_columns(0, 4), Some(BE::Column(4)));
                assert_eq!(m.swap_columns(10, 2), Some(BE::Column(10)));
            }

            #[test]
            fn swaps_columns() {
                let mut m = Matrix::<u8>::new_identity(4).unwrap();

                assert_eq!(m.swap_columns(0, 2), None);
                assert_eq!(m.data, vec![0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]);
            }
        }
    }

    mod operations {
        use super::*;

        #[test]
        fn scale() {
            let mut m =
                Matrix::new_from_data(&vec![vec![1, 2, 3], vec![2, 3, 1], vec![3, 1, 2]]).unwrap();
            m.scale(2);

            assert_eq!(m.data, vec![2, 4, 6, 4, 6, 2, 6, 2, 4]);
        }

        mod scale_row {
            use super::*;

            #[test]
            fn handles_errors() {
                let mut m = Matrix::new_from_data(&vec![vec![1]]).unwrap();

                assert_eq!(m.scale_row(4, 3), Some(BE::Row(4)));
            }

            #[test]
            fn scales_row() {
                let mut m = Matrix::<u128>::new_identity(2).unwrap();

                assert_eq!(m.scale_row(0, 3), None);
                assert_eq!(m.data, vec![3, 0, 0, 1]);
            }
        }

        mod add_scaled_row {
            use super::*;

            #[test]
            fn handles_errors() {
                let mut m = Matrix::new_from_data(&vec![vec![1, 2], vec![2, 1]]).unwrap();

                assert_eq!(m.add_scaled_row(2, 0, 3), Some(BE::Row(2)));
                assert_eq!(m.add_scaled_row(1, 5, 8), Some(BE::Row(5)));
            }

            #[test]
            fn adds_scaled_row() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![1, 0, 1], vec![2, 3, 0], vec![0, 5, 9]])
                        .unwrap();

                assert_eq!(m.add_scaled_row(0, 2, 3), None);
                assert_eq!(m.data, vec![1, 0, 1, 2, 3, 0, 3, 5, 12]);
            }
        }

        mod scale_column {
            use super::*;

            #[test]
            fn handles_errors() {
                let mut m = Matrix::new_from_data(&vec![vec![1]]).unwrap();

                assert_eq!(m.scale_column(4, 3), Some(BE::Column(4)));
            }

            #[test]
            fn scales_column() {
                let mut m = Matrix::<u128>::new_identity(2).unwrap();

                assert_eq!(m.scale_column(0, 3), None);
                assert_eq!(m.data, vec![3, 0, 0, 1]);
            }
        }

        mod add_scaled_column {
            use super::*;

            #[test]
            fn handles_errors() {
                let mut m = Matrix::new_from_data(&vec![vec![1, 2], vec![2, 1]]).unwrap();

                assert_eq!(m.add_scaled_column(2, 0, 3), Some(BE::Column(2)));
                assert_eq!(m.add_scaled_column(1, 5, 8), Some(BE::Column(5)));
            }

            #[test]
            fn adds_scaled_column() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![1, 0, 1], vec![2, 3, 0], vec![0, 5, 9]])
                        .unwrap();

                assert_eq!(m.add_scaled_column(0, 2, 3), None);
                assert_eq!(m.data, vec![1, 0, 4, 2, 3, 6, 0, 5, 9]);
            }
        }

        mod resize {
            use super::*;

            #[test]
            fn handles_errors() {
                let mut m = Matrix::new_from_data(&vec![vec![2, 5, 3], vec![9, 1, 6]]).unwrap();

                assert_eq!(m.resize((0, 5)), Some(BE::Row(0)));
                assert_eq!(m.resize((2, 0)), Some(BE::Column(0)));
                assert_eq!(m.resize((3, 4)), Some(BE::Both(3, 4)));
            }

            #[test]
            fn resizes() {
                let mut m = Matrix::new_from_data(&vec![vec![2, 5, 3], vec![9, 1, 6]]).unwrap();

                assert_eq!(m.resize((3, 2)), None);
                assert_eq!(m.rows, 3);
                assert_eq!(m.columns, 2);

                assert_eq!(m.resize((6, 1)), None);
                assert_eq!(m.rows, 6);
                assert_eq!(m.columns, 1);

                assert_eq!(m.resize((1, 6)), None);
                assert_eq!(
                    m,
                    Matrix {
                        data: vec![2, 5, 3, 9, 1, 6],
                        rows: 1,
                        columns: 6
                    }
                );
            }
        }

        mod remove_row {
            use super::*;

            #[test]
            fn handles_errors() {
                let mut m = Matrix::<u64>::new(10, 7).unwrap();

                assert_eq!(m.remove_row(19), Some(19));
                assert_eq!(m.remove_row(10), Some(10));
            }

            #[test]
            fn removes_row() {
                let mut m = Matrix::new_from_data(&vec![
                    vec![1, 2, 3],
                    vec![4, 5, 6],
                    vec![7, 8, 9],
                    vec![10, 11, 10],
                    vec![9, 8, 7],
                    vec![6, 5, 4],
                    vec![3, 2, 1],
                ])
                .unwrap();

                assert_eq!(m.remove_row(2), None);
                assert_eq!(
                    m,
                    Matrix {
                        data: vec![1, 2, 3, 4, 5, 6, 10, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                        rows: 6,
                        columns: 3
                    }
                );
                assert_eq!(m.remove_row(5), None);
                assert_eq!(
                    m,
                    Matrix {
                        data: vec![1, 2, 3, 4, 5, 6, 10, 11, 10, 9, 8, 7, 6, 5, 4],
                        rows: 5,
                        columns: 3
                    }
                );
            }
        }

        mod remove_column {
            use super::*;

            #[test]
            fn handles_errors() {
                let mut m = Matrix::new_with_data(
                    5,
                    vec![3, 5, 4, 6, 8, 6, 3, 2, 1, 6, 8, 5, 8, 4, 5, 6, 7, 3, 4, 0],
                )
                .unwrap();

                assert_eq!(m.remove_column(0), None);
                assert_eq!(
                    m,
                    Matrix {
                        data: vec![5, 4, 6, 8, 3, 2, 1, 6, 5, 8, 4, 5, 7, 3, 4, 0],
                        rows: 4,
                        columns: 4
                    }
                );
                assert_eq!(m.remove_column(2), None);
                assert_eq!(
                    m,
                    Matrix {
                        data: vec![5, 4, 8, 3, 2, 6, 5, 8, 5, 7, 3, 0],
                        rows: 4,
                        columns: 3
                    }
                );
            }
        }
    }

    mod derivers {
        use super::*;

        #[test]
        fn transpose() {
            let m1 = Matrix::new_from_data(&vec![vec![1, 4, 2], vec![9, 7, 1], vec![4, 6, 2]])
                .unwrap()
                .transpose();
            let m2 =
                Matrix::new_from_data(&vec![vec![1, 9, 4], vec![4, 7, 6], vec![2, 1, 2]]).unwrap();

            assert_eq!(m1, m2);
        }

        mod minor {
            use super::*;

            #[test]
            fn handles_errors() {
                let m1 = Matrix::<i32>::new(5, 6).unwrap();
                let m2 = Matrix::<i32>::new_identity(5).unwrap();

                assert_eq!(m1.minor((3, 2)), Err(ME::SizingError(SE::NotSquare)));
                assert_eq!(m1.minor((5, 7)), Err(ME::SizingError(SE::NotSquare)));
                assert_eq!(m2.minor((7, 0)), Err(ME::BoundsError(BE::Row(7))));
                assert_eq!(m2.minor((1, 5)), Err(ME::BoundsError(BE::Column(5))));
                assert_eq!(m2.minor((6, 9)), Err(ME::BoundsError(BE::Both(6, 9))));
            }

            #[test]
            fn gets_minor() {
                let m1 = Matrix::new_from_data(&vec![vec![3, 2], vec![1, 7]]).unwrap();
                let m2 = Matrix::<u128>::new_identity(3).unwrap();

                assert_eq!(m1.minor((1, 1)).unwrap(), 3);
                assert_eq!(m2.minor((0, 0)).unwrap(), 1);
            }
        }

        mod minor_matrix {
            use super::*;

            #[test]
            fn handles_errors() {
                let m = Matrix::new_from_data(&vec![vec![1, 2, 31]]).unwrap();

                assert_eq!(m.minor_matrix(), None);
            }

            #[test]
            fn gets_minor_matrix() {
                let m1 = Matrix::<u64>::new_from_data(&vec![vec![5, 3], vec![2, 9]]).unwrap();
                let m2 = Matrix::<i32>::new_identity(4).unwrap();

                assert_eq!(m1.minor_matrix().unwrap().data, vec![9, 2, 3, 5]);
                assert_eq!(m2.minor_matrix().unwrap(), m2);
            }
        }

        mod cofacter {
            use super::*;

            #[test]
            fn handles_errors() {
                let m1 = Matrix::<i16>::new(5, 2).unwrap();
                let m2 = Matrix::new_from_data(&vec![vec![1, 2, 3], vec![3, 2, 1]]).unwrap();

                assert_eq!(m1.cofactor(), None);
                assert_eq!(m2.cofactor(), None);
            }

            #[test]
            fn creates_cofacter() {
                let m1 = Matrix::new_from_data(&vec![
                    vec![1, 2, 3, 4],
                    vec![5, 6, 7, 8],
                    vec![8, 7, 6, 5],
                    vec![4, 3, 2, 1],
                ])
                .unwrap();
                let m2 = Matrix::<i8>::new_identity(5).unwrap();
                let m3 = Matrix::new_from_data(&vec![
                    vec![2, 5, 0, 8, 4],
                    vec![10, 2, 7, 2, 0],
                    vec![8, 5, 1, 0, 6],
                    vec![8, 8, 3, 3, 3],
                    vec![5, 2, 0, 5, 9],
                ])
                .unwrap();

                assert_eq!(m1.cofactor().unwrap().data, vec![0; 16]);
                assert_eq!(m2.cofactor().unwrap().data, m2.data);
                assert_eq!(
                    m3.cofactor().unwrap().data,
                    vec![
                        -1578, 564, 2346, -885, 1243, -610, 376, 498, -291, 417, -2234, 752, 3154,
                        -621, 1419, 2168, -1128, -3028, 886, -1446, 1468, -376, -2136, 512, -1288
                    ]
                );
            }
        }

        mod adjunct {
            use super::*;

            #[test]
            fn handles_errors() {
                let m1 = Matrix::<i16>::new(5, 2).unwrap();
                let m2 = Matrix::new_from_data(&vec![vec![1, 2, 3], vec![3, 2, 1]]).unwrap();

                assert_eq!(m1.adjunct(), None);
                assert_eq!(m2.adjunct(), None);
            }

            #[test]
            fn creates_adjunct() {
                let m1 = Matrix::new_from_data(&vec![vec![2, 7, 6], vec![3, 6, 9], vec![4, 8, 1]])
                    .unwrap();
                let m2 = Matrix::<i32>::new_identity(4).unwrap();

                assert_eq!(
                    m1.adjunct().unwrap().data,
                    vec![-66, 41, 27, 33, -22, 0, 0, 12, -9]
                );
                assert_eq!(m2.adjunct().unwrap(), m2);
            }
        }

        mod determinant {
            use super::*;

            #[test]
            fn handles_errors() {
                let m = Matrix::<i32>::new(7, 5).unwrap();

                assert_eq!(m.determinant(), None);
            }

            #[test]
            fn derives_determinant() {
                let m1 = Matrix::<u32>::new(4, 4).unwrap();
                let m2 = Matrix::new_from_data(&vec![vec![1, 2], vec![5, 7]]).unwrap();
                let m3 = Matrix::<i16>::new_identity(9).unwrap();

                assert_eq!(m1.determinant(), Some(0));
                assert_eq!(m2.determinant(), Some(-3));
                assert_eq!(m3.determinant(), Some(1));
            }
        }

        mod inverse {
            use super::*;

            #[test]
            fn handles_errors() {
                let m1 = Matrix::new_from_data(&vec![vec![1, 4], vec![2, 8]]).unwrap();
                let m2 = Matrix::<i16>::new(4, 5).unwrap();

                assert_eq!(m1.inverse(), Err(IE::InvalidDeterminant));
                assert_eq!(m2.inverse(), Err(IE::NotSquare));
            }

            #[test]
            fn derives_inverse() {
                let m1 = Matrix::new_from_data(&vec![vec![1, 0, 2], vec![0, 4, 1], vec![0, 1, 0]])
                    .unwrap();
                let m2 = Matrix::<i8>::new_identity(6).unwrap();
                let m3 =
                    Matrix::new_from_data(&vec![vec![1, -1, 1], vec![2, 3, 0], vec![0, -2, 1]])
                        .unwrap();

                assert_eq!(
                    m1.inverse(),
                    Ok(Matrix {
                        data: vec![1, -2, 8, 0, 0, 1, 0, 1, -4],
                        rows: 3,
                        columns: 3
                    })
                );
                assert_eq!(m2.inverse(), Ok(m2));
                assert_eq!(
                    m3.inverse().unwrap().data,
                    vec![3, -1, -3, -2, 1, 2, -4, 2, 5]
                );
            }
        }

        mod fast_inverse {
            use super::*;

            #[test]
            fn handles_errors() {
                let m1 = Matrix::new_from_data(&vec![vec![1, 4], vec![2, 8]]).unwrap();
                let m2 = Matrix::<i16>::new(4, 5).unwrap();

                assert_eq!(m1.fast_inverse(), Err(IE::InvalidDeterminant));
                assert_eq!(m2.fast_inverse(), Err(IE::NotSquare));
            }

            #[test]
            fn derives_inverse() {
                let m1 = Matrix::new_from_data(&vec![
                    vec![1f32, 0f32, 2f32],
                    vec![0f32, 4f32, 1f32],
                    vec![0f32, 1f32, 0f32],
                ])
                .unwrap();
                let m2 = Matrix::<i8>::new_identity(6).unwrap();
                let m3 = Matrix::new_from_data(&vec![
                    vec![1f32, -1f32, 1f32],
                    vec![2f32, 3f32, 0f32],
                    vec![0f32, -2f32, 1f32],
                ])
                .unwrap();

                assert_eq!(
                    m1.fast_inverse(),
                    Ok(Matrix {
                        data: vec![1f32, -2f32, 8f32, 0f32, 0f32, 1f32, 0f32, 1f32, -4f32],
                        rows: 3,
                        columns: 3
                    })
                );
                assert_eq!(m2.fast_inverse(), Ok(m2));
                assert_eq!(
                    m3.fast_inverse()
                        .unwrap()
                        .data
                        .iter()
                        .map(|f| f.round())
                        .collect::<Vec<_>>(),
                    vec![3f32, -1f32, -3f32, -2f32, 1f32, 2f32, -4f32, 2f32, 5f32]
                );
            }
        }

        mod as_resize {
            use super::*;

            #[test]
            fn handles_errors() {
                let m = Matrix::new_from_data(&vec![vec![2, 5, 3], vec![9, 1, 6]]).unwrap();

                assert_eq!(m.as_resize((0, 5)), Err(BE::Row(0)));
                assert_eq!(m.as_resize((2, 0)), Err(BE::Column(0)));
                assert_eq!(m.as_resize((3, 4)), Err(BE::Both(3, 4)));
            }

            #[test]
            fn resizes() {
                let m1 = Matrix::new_from_data(&vec![vec![2, 5, 3], vec![9, 1, 6]]).unwrap();
                let m2 = m1.as_resize((3, 2)).unwrap();
                let m3 = m1.as_resize((6, 1)).unwrap();
                let m4 = m1.as_resize((1, 6)).unwrap();

                assert_eq!(
                    m2,
                    Matrix {
                        data: vec![2, 5, 3, 9, 1, 6],
                        rows: 3,
                        columns: 2
                    }
                );
                assert_eq!(
                    m3,
                    Matrix {
                        data: vec![2, 5, 3, 9, 1, 6],
                        rows: 6,
                        columns: 1
                    }
                );
                assert_eq!(
                    m4,
                    Matrix {
                        data: vec![2, 5, 3, 9, 1, 6],
                        rows: 1,
                        columns: 6
                    }
                );
            }
        }
    }

    mod traits {
        use super::*;

        #[test]
        fn debug() {
            let m1 = Matrix::<u8>::new(1, 1).unwrap();
            let m2 = Matrix::<i16>::new(1, 7).unwrap();
            let m3 = Matrix::<u32>::new(5, 1).unwrap();
            let m4 = Matrix::<i8>::new(3, 8).unwrap();

            assert_eq!("[0]", format!("{:?}", m1));
            assert_eq!("[0, 0, 0, 0, 0, 0, 0]", format!("{:?}", m2));
            assert_eq!("[0]\n[0]\n[0]\n[0]\n[0]", format!("{:?}", m3));
            assert_eq!(
                "[0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0]\n[0, 0, 0, 0, 0, 0, 0, 0]",
                format!("{:?}", m4)
            );
        }

        #[test]
        fn index() {
            let m = Matrix::new_from_data(&vec![
                vec![2, 5, 7, 2, 1],
                vec![8, 0, 5, 3, 6],
                vec![6, 4, 3, 6, 7],
                vec![1, 7, 9, 3, 4],
            ])
            .unwrap();

            assert_eq!(m[(3, 4)], 4);
            assert_eq!(m[(1, 2)], 5);
            assert_eq!(m[(3, 1)], 7);
            assert_eq!(m[(0, 3)], 2);
        }

        #[test]
        fn index_mut() {
            let mut m = Matrix::<u8>::new(3, 2).unwrap();

            m[(1, 0)] = 5;
            m[(2, 1)] = 3;
            m[(0, 0)] = 1;
            assert_eq!(m.data, vec![1, 0, 5, 0, 0, 3]);
        }

        #[test]
        fn add_assign() {
            let mut m1 = Matrix::<i8>::new(4, 4).unwrap();
            let m2 = Matrix::<i8>::new_identity(4).unwrap();
            let m3 = Matrix::new_from_data(&vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![8, 7, 6, 5],
                vec![4, 3, 2, 1],
            ])
            .unwrap();

            m1 += m2;
            assert_eq!(
                m1.data,
                vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            );
            m1 += m3;
            assert_eq!(
                m1.data,
                vec![2, 2, 3, 4, 5, 7, 7, 8, 8, 7, 7, 5, 4, 3, 2, 2]
            );
        }

        mod add {
            use super::*;

            #[test]
            fn handles_errors() {
                let m1 = Matrix::<u8>::new(4, 4).unwrap();
                let m2 = Matrix::<u8>::new(3, 4).unwrap();
                let m3 = Matrix::<u8>::new(4, 5).unwrap();
                let m4 = Matrix::<u8>::new(3, 5).unwrap();

                assert_eq!(&m1 + &m2, Err(SE::Row(3)));
                assert_eq!(&m1 + &m3, Err(SE::Column(5)));
                assert_eq!(m1 + m4, Err(SE::Both(3, 5)));
            }

            #[test]
            fn adds() {
                let m = Matrix::<i8>::new_identity(4).unwrap();

                assert_eq!(
                    (&m + &Matrix::<i8>::new(4, 4).unwrap()).unwrap().data,
                    vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
                );
                assert_eq!(
                    (m + Matrix::new_from_data(&vec![
                        vec![1, 2, 3, 4],
                        vec![5, 6, 7, 8],
                        vec![8, 7, 6, 5],
                        vec![4, 3, 2, 1],
                    ])
                    .unwrap())
                    .unwrap()
                    .data,
                    vec![2, 2, 3, 4, 5, 7, 7, 8, 8, 7, 7, 5, 4, 3, 2, 2]
                );
            }
        }

        #[test]
        fn sub_assign() {
            let m1 = Matrix::<i8>::new(4, 4).unwrap();
            let mut m2 = Matrix::<i8>::new_identity(4).unwrap();
            let m3 = Matrix::new_from_data(&vec![
                vec![1, 2, 3, 4],
                vec![5, 6, 7, 8],
                vec![8, 7, 6, 5],
                vec![4, 3, 2, 1],
            ])
            .unwrap();

            m2 -= m1;
            assert_eq!(
                m2.data,
                vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
            );
            m2 -= m3;
            assert_eq!(
                m2.data,
                vec![0, -2, -3, -4, -5, -5, -7, -8, -8, -7, -5, -5, -4, -3, -2, 0]
            );
        }

        mod sub {
            use super::*;

            #[test]
            fn handles_errors() {
                let m1 = Matrix::<u8>::new(4, 4).unwrap();
                let m2 = Matrix::<u8>::new(3, 4).unwrap();
                let m3 = Matrix::<u8>::new(4, 5).unwrap();
                let m4 = Matrix::<u8>::new(3, 5).unwrap();

                assert_eq!(&m1 - &m2, Err(SE::Row(3)));
                assert_eq!(&m1 - &m3, Err(SE::Column(5)));
                assert_eq!(m1 - m4, Err(SE::Both(3, 5)));
            }

            #[test]
            fn subs() {
                let m = Matrix::<i8>::new_identity(4).unwrap();

                assert_eq!(
                    (&m - &Matrix::<i8>::new(4, 4).unwrap()).unwrap().data,
                    vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
                );
                assert_eq!(
                    (m - Matrix::new_from_data(&vec![
                        vec![1, 2, 3, 4],
                        vec![5, 6, 7, 8],
                        vec![8, 7, 6, 5],
                        vec![4, 3, 2, 1],
                    ])
                    .unwrap())
                    .unwrap()
                    .data,
                    vec![0, -2, -3, -4, -5, -5, -7, -8, -8, -7, -5, -5, -4, -3, -2, 0]
                );
            }
        }

        mod mul {
            use super::*;

            #[test]
            fn handles_errors() {
                let m1 = Matrix::<u16>::new(2, 5).unwrap();
                let m2 = Matrix::<u16>::new(4, 2).unwrap();

                assert_eq!(m1 * m2, Err(SE::Row(4)));
            }

            #[test]
            fn muls() {
                let m1 = Matrix::new_from_data(&vec![
                    vec![1, 6, 3],
                    vec![3, 7, 2],
                    vec![5, 4, 8],
                    vec![5, 6, 9],
                ])
                .unwrap();
                let m2 = Matrix::new_from_data(&vec![
                    vec![2, 7, 5, 7],
                    vec![9, 1, 8, 3],
                    vec![2, 4, 6, 5],
                ])
                .unwrap();
                let m3 = Matrix::<i8>::new_identity(3).unwrap();

                assert_eq!(
                    &m1 * &m2,
                    Ok(Matrix {
                        data: vec![
                            62, 25, 71, 40, 73, 36, 83, 52, 62, 71, 105, 87, 82, 77, 127, 98
                        ],
                        rows: 4,
                        columns: 4
                    })
                );
                assert_eq!((&m1 * &m3).unwrap(), m1);
            }
        }

        #[test]
        fn neg() {
            let m1 = Matrix::<i32>::new_identity(3).unwrap();

            assert_eq!(m1.neg().data, vec![-1, 0, 0, 0, -1, 0, 0, 0, -1]);
        }
    }
}
