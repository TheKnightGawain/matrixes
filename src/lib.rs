extern crate num;

use num::{One, Zero};
use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Neg, Range, Sub},
};

#[derive(Debug, Clone, PartialEq, Eq)]
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
    pub fn new(rows: &usize, columns: &usize) -> Result<Self, usize>
    where
        T: Zero + Copy,
    {
        if *rows == 0 {
            return Err(*rows);
        }

        if *columns == 0 {
            return Err(*columns);
        }

        Ok(Self {
            data: vec![T::zero(); rows * columns],
            rows: *rows,
            columns: *columns,
        })
    }

    pub fn new_identity(n: &usize) -> Result<Self, usize>
    where
        T: Zero + One,
    {
        if *n == 0 {
            return Err(*n);
        }

        Ok(Self {
            data: (0..n.pow(2))
                .map(|i| {
                    if i % (*n + 1) == 0 {
                        T::one()
                    } else {
                        T::zero()
                    }
                })
                .collect(),
            rows: *n,
            columns: *n,
        })
    }
    /// data is in row-major order.
    pub fn new_with_data(rows: &usize, columns: &usize, data: &Vec<T>) -> Result<Self, usize> {
        if *rows == 0 {
            return Err(*rows);
        }

        if *columns == 0 {
            return Err(*columns);
        }

        if data.len() != rows * columns {
            return Err(data.len());
        }

        Ok(Self {
            data: data.to_owned(),
            rows: *rows,
            columns: *columns,
        })
    }

    /// data is vec of rows of elemnts
    pub fn new_from_data(data: &Vec<Vec<T>>) -> Result<Self, usize>
    where
        T: Copy,
    {
        let rows = data.len();

        if rows == 0 {
            return Err(rows);
        }

        let columns = data[0].len();
        let mut elements: Vec<T> = Vec::new();

        for r in data.iter() {
            if r.len() != columns || r.len() == 0 {
                return Err(r.len());
            }

            for &e in r.iter() {
                elements.push(e);
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
    pub fn data(&self) -> Vec<T> {
        self.data.clone()
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn get_row(&self, i: &usize) -> Result<Vec<T>, usize> {
        if *i >= self.rows {
            return Err(*i);
        }

        Ok((0..self.columns)
            .map(|c| self.get_element((i, &c)).unwrap())
            .collect())
    }

    pub fn get_rows(&self, range: Range<usize>) -> Result<Vec<Vec<T>>, usize> {
        if range.end > self.rows {
            return Err(range.end);
        }

        Ok(range.map(|r| self.get_row(&r).unwrap()).collect())
    }

    pub fn get_column(&self, i: &usize) -> Result<Vec<T>, usize> {
        if *i >= self.columns {
            return Err(*i);
        }

        Ok((0..self.rows)
            .map(|r| self.get_element((&r, i)).unwrap())
            .collect())
    }

    pub fn get_columns(&self, range: Range<usize>) -> Result<Vec<Vec<T>>, usize> {
        if range.end > self.columns {
            return Err(range.end);
        }

        Ok(range.map(|c| self.get_column(&c).unwrap()).collect())
    }

    pub fn get_element(&self, element: (&usize, &usize)) -> Result<T, usize> {
        if *element.0 >= self.rows {
            return Err(*element.0);
        }

        if *element.1 >= self.columns {
            return Err(*element.1);
        }

        Ok(self.data[element.0 * self.columns + element.1])
    }
}

// setters
impl<T> Matrix<T>
where
    T: Copy,
{
    pub fn set_element(
        &mut self,
        element: (&usize, &usize),
        value: &T,
    ) -> Result<&mut Self, usize> {
        if *element.0 >= self.rows {
            return Err(*element.0);
        }

        if *element.1 >= self.rows {
            return Err(*element.1);
        }

        self.data[element.0 * self.columns + element.1] = *value;
        Ok(self)
    }

    pub fn set_row(&mut self, row: &usize, values: &Vec<T>) -> Result<&mut Self, usize> {
        if *row >= self.rows {
            return Err(*row);
        }

        if values.len() != self.columns {
            return Err(values.len());
        }

        for c in 0..self.columns {
            self.data[row * self.rows + c] = values[c];
        }
        Ok(self)
    }

    pub fn set_rows(
        &mut self,
        rows: &Range<usize>,
        values: &Vec<Vec<T>>,
    ) -> Result<&mut Self, usize> {
        if rows.end > self.rows {
            return Err(rows.end);
        }

        if values.len() > self.rows || values.len() != rows.len() {
            return Err(values.len());
        }

        for (count, r) in rows.to_owned().enumerate() {
            match self.set_row(&r, &values[count]) {
                Err(e) => return Err(e),
                _ => (),
            }
        }
        Ok(self)
    }

    pub fn set_column(&mut self, column: &usize, values: &Vec<T>) -> Result<&mut Self, usize> {
        if *column >= self.columns {
            return Err(*column);
        }

        if values.len() != self.rows {
            return Err(values.len());
        }

        for r in 0..self.rows {
            self.data[r * self.rows + column] = values[r];
        }
        Ok(self)
    }

    pub fn set_columns(
        &mut self,
        columns: &Range<usize>,
        values: &Vec<Vec<T>>,
    ) -> Result<&mut Self, usize> {
        if columns.end > self.columns {
            return Err(columns.end);
        }

        if values.len() > self.columns || values.len() < columns.len() {
            return Err(values.len());
        }

        for (count, c) in columns.to_owned().enumerate() {
            match self.set_column(&c, &values[count]) {
                Err(e) => return Err(e),
                _ => (),
            };
        }
        Ok(self)
    }
}

// swappers
impl<T> Matrix<T>
where
    T: Copy,
{
    pub fn swap_elements(
        &mut self,
        el1: (&usize, &usize),
        el2: (&usize, &usize),
    ) -> Result<&mut Self, usize> {
        if *el1.0 >= self.rows {
            return Err(*el1.0);
        }

        if *el1.1 >= self.columns {
            return Err(*el1.1);
        }

        if *el2.0 >= self.rows {
            return Err(*el2.0);
        }

        if *el2.1 >= self.columns {
            return Err(*el2.1);
        }

        let temp = match self.get_element(el1) {
            Ok(el) => el,
            Err(e) => return Err(e),
        };
        match self.set_element(
            el1,
            &match self.get_element(el2) {
                Ok(el) => el,
                Err(e) => return Err(e),
            },
        ) {
            Err(e) => return Err(e),
            _ => (),
        };
        match self.set_element(el2, &temp) {
            Err(e) => return Err(e),
            _ => (),
        };
        Ok(self)
    }

    pub fn swap_rows(&mut self, row1: &usize, row2: &usize) -> Result<&mut Self, usize> {
        if *row1 >= self.rows {
            return Err(*row1);
        }

        if *row2 >= self.rows {
            return Err(*row2);
        }

        let temp = match self.get_row(row1) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        match self.set_row(
            row1,
            &match self.get_row(row2) {
                Ok(v) => v,
                Err(e) => return Err(e),
            },
        ) {
            Err(e) => return Err(e),
            _ => (),
        };
        match self.set_row(row2, &temp) {
            Err(e) => return Err(e),
            _ => (),
        };
        Ok(self)
    }

    pub fn swap_columns(&mut self, col1: &usize, col2: &usize) -> Result<&mut Self, usize> {
        if *col1 >= self.columns {
            return Err(*col1);
        }

        if *col2 >= self.columns {
            return Err(*col2);
        }

        let temp = match self.get_column(col1) {
            Ok(v) => v,
            Err(e) => return Err(e),
        };
        match self.set_column(
            col1,
            &match self.get_column(col2) {
                Ok(v) => v,
                Err(e) => return Err(e),
            },
        ) {
            Err(e) => return Err(e),
            _ => (),
        };
        match self.set_column(col2, &temp) {
            Err(e) => return Err(e),
            _ => (),
        }
        Ok(self)
    }
}

// operations
impl<T> Matrix<T>
where
    T: Copy + Mul<Output = T>,
{
    pub fn scale(&mut self, factor: &T) -> &mut Self {
        self.data.iter_mut().for_each(|e| *e = *e * *factor);
        self
    }

    pub fn scale_row(&mut self, row: &usize, factor: &T) -> Result<&mut Self, usize> {
        if *row >= self.rows {
            return Err(*row);
        }

        match self.set_row(
            row,
            &match self.get_row(row) {
                Ok(r) => r,
                Err(e) => return Err(e),
            }
            .iter()
            .map(|&e| e * *factor)
            .collect(),
        ) {
            Ok(_) => Ok(self),
            Err(e) => Err(e),
        }
    }

    pub fn add_scaled_row(
        &mut self,
        from: &usize,
        to: &usize,
        factor: &T,
    ) -> Result<&mut Self, usize>
    where
        T: Add<Output = T>,
    {
        if *from >= self.rows {
            return Err(*from);
        }

        if *to >= self.rows {
            return Err(*to);
        }

        match self.set_row(
            to,
            &(match self.get_row(to) {
                Ok(v) => v,
                Err(e) => return Err(e),
            }
            .iter()
            .zip(
                match self.get_row(from) {
                    Ok(v) => v,
                    Err(e) => return Err(e),
                }
                .iter(),
            )
            .map(|(&to, &from)| to + (from * *factor))
            .collect::<Vec<T>>()),
        ) {
            Err(e) => return Err(e),
            _ => (),
        };

        Ok(self)
    }

    pub fn scale_column(&mut self, column: &usize, factor: &T) -> Result<&mut Self, usize> {
        if *column >= self.columns {
            return Err(*column);
        }

        match self.set_column(
            column,
            &match self.get_column(column) {
                Ok(v) => v,
                Err(e) => return Err(e),
            }
            .iter()
            .map(|&e| e * *factor)
            .collect(),
        ) {
            Ok(_) => Ok(self),
            Err(e) => Err(e),
        }
    }

    pub fn add_scaled_column(
        &mut self,
        from: &usize,
        to: &usize,
        factor: &T,
    ) -> Result<&mut Self, usize>
    where
        T: Add<Output = T>,
    {
        if *from >= self.columns {
            return Err(*from);
        }

        if *to >= self.columns {
            return Err(*to);
        }

        match self.set_column(
            to,
            &match self.get_column(to) {
                Ok(v) => v,
                Err(e) => return Err(e),
            }
            .iter()
            .zip(
                match self.get_column(from) {
                    Ok(v) => v,
                    Err(e) => return Err(e),
                }
                .iter(),
            )
            .map(|(&to, &from)| to + (from * *factor))
            .collect(),
        ) {
            Err(e) => return Err(e),
            _ => (),
        };

        Ok(self)
    }
}

// derivers
impl<T> Matrix<T>
where
    T: Copy,
{
    pub fn transpose(&self) -> Self {
        Matrix::new_from_data(&self.get_columns(0..self.columns).unwrap()).unwrap()
    }

    pub fn minor(&self, element: (&usize, &usize)) -> Result<T, (usize, usize)>
    where
        T: Neg<Output = T> + Mul<Output = T> + Zero,
    {
        if self.rows != self.columns {
            return Err((self.rows, self.columns));
        }

        if *element.0 >= self.rows || *element.1 >= self.columns {
            return Err((*element.0, *element.1));
        }

        let mut data: Vec<Vec<T>> = Vec::new();

        for r in self.get_rows(0..*element.0).unwrap() {
            data.push({
                let left_slice = &r[..*element.1];
                let right_slice = &r[element.1 + 1..];
                let mut out = left_slice.to_vec();
                out.append(&mut right_slice.to_vec());
                out
            });
        }

        for r in self.get_rows(element.0 + 1..self.rows).unwrap() {
            data.push({
                let left_slice = &r[..*element.1];
                let right_slice = &r[element.1 + 1..];
                let mut out = left_slice.to_vec();
                out.append(&mut right_slice.to_vec());
                out
            })
        }

        match Matrix::new_from_data(&data) {
            Ok(m) => m.determinant(),
            Err(_) => Err((*element.0, *element.1)),
        }
    }

    pub fn minor_matrix(&self) -> Result<Self, (usize, usize)>
    where
        T: Neg<Output = T> + Mul<Output = T> + Zero,
    {
        if self.rows != self.columns {
            return Err((self.rows, self.columns));
        }

        let mut data: Vec<T> = Vec::new();

        for r in 0..self.rows {
            for c in 0..self.columns {
                data.push(match self.minor((&r, &c)) {
                    Ok(e) => e,
                    Err(_) => return Err((self.rows, self.columns)),
                });
            }
        }

        match Matrix::new_with_data(&self.rows, &self.columns, &data) {
            Ok(m) => Ok(m),
            Err(_) => Err((self.rows, self.columns)),
        }
    }

    pub fn cofactor(&self) -> Self
    where
        T: Neg<Output = T>,
    {
        Matrix::new_from_data(
            &self
                .get_rows(0..self.rows)
                .unwrap()
                .iter()
                .enumerate()
                .map(|(r, row)| {
                    row.iter()
                        .enumerate()
                        .map(|(c, &e)| if (r + c) % 2 == 0 { e } else { e.neg() })
                        .collect()
                })
                .collect(),
        )
        .unwrap()
    }

    pub fn adjoint(&self) -> Result<Self, (usize, usize)>
    where
        T: Neg<Output = T> + Mul<Output = T> + Zero,
    {
        if self.rows != self.columns {
            return Err((self.rows, self.columns));
        }

        Ok(match self.transpose().minor_matrix() {
            Ok(m) => m,
            Err(i) => return Err(i),
        }
        .cofactor())
    }

    pub fn determinant(&self) -> Result<T, (usize, usize)>
    where
        T: Neg<Output = T> + Mul<Output = T> + Zero,
    {
        if self.rows != self.columns {
            return Err((self.rows, self.columns));
        }

        if self.rows == 1 {
            return Ok(self.data[0]);
        }

        Ok(self
            .get_row(&0)
            .unwrap()
            .iter()
            .enumerate()
            .fold(T::zero(), |res, (c, &e)| {
                let det = e * self.minor((&0, &c)).unwrap();
                res + if c % 2 == 0 { det } else { det.neg() }
            }))
    }

    pub fn inverse(&self) -> Result<Self, (usize, usize)>
    where
        T: Neg<Output = T> + Mul<Output = T> + Zero + One + Div<Output = T>,
    {
        if self.rows != self.columns {
            return Err((self.rows, self.columns));
        }

        let mut out = match self.adjoint() {
            Ok(m) => m,
            Err(e) => return Err(e),
        };

        out.scale(
            &(T::one()
                / match self.determinant() {
                    Ok(d) => d,
                    Err(e) => return Err(e),
                }),
        );

        Ok(out)
    }
}

impl<T> Add for Matrix<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = Result<Self, (usize, usize)>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.rows != rhs.rows {
            return Err((self.rows, rhs.rows));
        }

        if self.columns != rhs.columns {
            return Err((self.columns, rhs.columns));
        }

        let mut result = self;
        result
            .data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(s, &o)| *s = *s + o);
        Ok(result)
    }
}

impl<T> Sub for Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Result<Self, (usize, usize)>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.rows != rhs.rows {
            return Err((self.rows, rhs.rows));
        }

        if self.columns != rhs.columns {
            return Err((self.columns, rhs.columns));
        }

        let mut result = self;
        result
            .data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(s, &o)| *s = *s - o);
        Ok(result)
    }
}

impl<T> Mul for Matrix<T>
where
    T: Zero + Copy + Mul<Output = T> + One,
{
    type Output = Result<Self, (usize, usize)>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.columns != rhs.rows {
            return Err((self.columns, rhs.rows));
        }

        match Matrix::new_from_data(
            &(0..self.rows)
                .map(|r| {
                    (0..rhs.columns)
                        .map(|c| {
                            self.get_row(&r)
                                .unwrap()
                                .iter()
                                .zip(rhs.get_column(&c).unwrap().iter())
                                .fold(T::zero(), |res, (&el, &er)| res + el * er)
                        })
                        .collect()
                })
                .collect(),
        ) {
            Ok(m) => Ok(m),
            Err(_) => return Err((self.rows, self.columns)),
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
            fn invalid_rows() {
                assert_eq!(Matrix::<u8>::new(&0, &3), Err(0))
            }

            #[test]
            fn invalid_columns() {
                assert_eq!(Matrix::<u8>::new(&2, &0), Err(0))
            }

            #[test]
            fn correct_data() {
                assert_eq!(Matrix::<u8>::new(&3, &2).unwrap(), {
                    Matrix {
                        rows: 3,
                        columns: 2,
                        data: vec![0, 0, 0, 0, 0, 0],
                    }
                });
                assert_eq!(Matrix::<u8>::new(&2, &3).unwrap(), {
                    Matrix {
                        rows: 2,
                        columns: 3,
                        data: vec![0, 0, 0, 0, 0, 0],
                    }
                });
            }
        }

        mod new_identity {
            use super::*;

            #[test]
            fn invalid_size() {
                assert_eq!(Matrix::<u8>::new_identity(&0), Err(0));
            }

            #[test]
            fn correct_data() {
                assert_eq!(Matrix::<u8>::new_identity(&3).unwrap(), {
                    Matrix {
                        rows: 3,
                        columns: 3,
                        data: vec![1, 0, 0, 0, 1, 0, 0, 0, 1],
                    }
                });
            }
        }

        mod new_with_data {
            use super::*;

            #[test]
            fn invalid_rows() {
                assert_eq!(Matrix::<u8>::new_with_data(&0, &3, &vec![]), Err(0));
            }

            #[test]
            fn invalid_columns() {
                assert_eq!(Matrix::<u8>::new_with_data(&5, &0, &vec![]), Err(0));
            }

            #[test]
            fn invalid_data() {
                assert_eq!(Matrix::new_with_data(&3, &3, &vec![1, 0, 0, 1]), Err(4));
            }

            #[test]
            fn correct_data() {
                assert_eq!(
                    Matrix::<u8>::new_with_data(&3, &3, &vec![0, 1, 2, 3, 4, 5, 6, 7, 8]).unwrap(),
                    {
                        Matrix {
                            rows: 3,
                            columns: 3,
                            data: vec![0, 1, 2, 3, 4, 5, 6, 7, 8],
                        }
                    }
                )
            }
        }

        mod new_from_data {
            use super::*;

            #[test]
            fn no_rows() {
                assert_eq!(Matrix::<u8>::new_from_data(&vec![]), Err(0));
            }

            #[test]
            fn no_columns() {
                assert_eq!(
                    Matrix::<u8>::new_from_data(&vec![vec![], vec![], vec![]]),
                    Err(0)
                );
            }

            #[test]
            fn inconsistant_length() {
                assert_eq!(
                    Matrix::<u8>::new_from_data(&vec![
                        vec![0, 1, 2, 3],
                        vec![10, 11, 12, 13],
                        vec![20],
                        vec![30, 31, 32, 33],
                    ]),
                    Err(1)
                );
            }

            #[test]
            fn derives_size() {
                assert_eq!(
                    Matrix::<u8>::new_from_data(&vec![vec![0, 1, 2], vec![3, 4, 5]]).unwrap(),
                    {
                        Matrix {
                            rows: 2,
                            columns: 3,
                            data: vec![0, 1, 2, 3, 4, 5],
                        }
                    }
                )
            }
        }
    }

    mod getters {
        use super::*;

        mod data {
            use super::*;

            #[test]
            fn correct_data() {
                let m = Matrix::new_from_data(&vec![
                    vec![0, 1, 2, 3],
                    vec![10, 11, 12, 13],
                    vec![20, 21, 22, 23],
                ])
                .unwrap();

                assert_eq!(m.data(), m.data);
            }
        }

        mod rows {
            use super::*;

            #[test]
            fn correct_data() {
                let m: Matrix<u8> = Matrix::new_identity(&3).unwrap();

                assert_eq!(m.rows(), m.rows);
            }
        }

        mod columns {
            use super::*;

            #[test]
            fn correct_data() {
                let m: Matrix<u8> = Matrix::new_identity(&3).unwrap();

                assert_eq!(m.columns, m.columns());
            }
        }

        mod get_element {
            use super::*;

            #[test]
            fn invalid_row() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.get_element((&3, &1)), Err(3));
            }

            #[test]
            fn invalid_column() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.get_element((&1, &5)), Err(5));
            }

            #[test]
            fn correct_data() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.get_element((&2, &1)).unwrap(), 21);
            }
        }

        mod get_row {
            use super::*;

            #[test]
            fn invalid_row() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.get_row(&4), Err(4));
            }

            #[test]
            fn correct_data() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.get_row(&1).unwrap(), vec![10, 11, 12]);
            }
        }

        mod get_rows {
            use super::*;

            #[test]
            fn empty_range() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.get_rows(1..1), Ok(Vec::<Vec<u8>>::new()));
            }

            #[test]
            fn invalid_row() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.get_rows(2..4), Err(4));
            }

            #[test]
            fn correct_data() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.get_rows(1..3).unwrap(),
                    vec![vec![10, 11, 12], vec![20, 21, 22]]
                )
            }
        }

        mod get_column {
            use super::*;

            #[test]
            fn invalid_column() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.get_column(&3), Err(3));
            }

            #[test]
            fn correct_data() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.get_column(&0).unwrap(), vec![0, 10, 20]);
            }
        }

        mod get_columns {
            use super::*;

            #[test]
            fn empty_range() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.get_columns(0..0).unwrap(), Vec::<Vec<u8>>::new());
            }

            #[test]
            fn invalid_column() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.get_columns(2..6), Err(6));
            }

            #[test]
            fn correct_data() {
                let m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.get_columns(0..m.columns).unwrap(),
                    vec![vec![0, 10, 20], vec![1, 11, 21], vec![2, 12, 22]]
                );
            }
        }
    }

    mod setters {
        use super::*;

        mod set_element {
            use super::*;

            #[test]
            fn invalid_row() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.set_element((&4, &1), &8), Err(4));
            }

            #[test]
            fn invalid_column() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.set_element((&0, &9), &3), Err(9));
            }

            #[test]
            fn correct_data() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.set_element((&1, &2), &7).unwrap().data,
                    vec![0, 1, 2, 10, 11, 7, 20, 21, 22]
                );
            }
        }

        mod set_row {
            use super::*;

            #[test]
            fn invalid_row() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.set_row(&5, &vec![3, 2, 1]), Err(5));
            }

            #[test]
            fn invalid_data() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.set_row(&1, &vec![1]), Err(1));
            }

            #[test]
            fn correct_data() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.set_row(&1, &vec![12, 11, 10]).unwrap().data,
                    vec![0, 1, 2, 12, 11, 10, 20, 21, 22]
                );
            }
        }

        mod set_rows {
            use super::*;

            #[test]
            fn empty_range() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.set_rows(&(1..1), &Vec::<Vec<u8>>::new()).unwrap().data,
                    vec![0, 1, 2, 10, 11, 12, 20, 21, 22]
                );
            }

            #[test]
            fn invalid_row() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.set_rows(&(3..4), &vec![vec![2, 1, 0]]), Err(4));
            }

            #[test]
            fn incorrect_length() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.set_rows(&(0..2), &vec![vec![4, 5], vec![9, 8, 7, 6]]),
                    Err(2)
                )
            }

            #[test]
            fn correct_data() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.set_rows(&(0..3), &vec![vec![1, 2, 3], vec![4, 5, 6], vec![0, 0, 0]])
                        .unwrap()
                        .data,
                    vec![1, 2, 3, 4, 5, 6, 0, 0, 0]
                );
            }
        }

        mod set_column {
            use super::*;

            #[test]
            fn invalid_column() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.set_column(&5, &vec![4, 4, 4]), Err(5));
            }

            #[test]
            fn incorrect_length() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.set_column(&1, &vec![1]), Err(1));
            }

            #[test]
            fn correct_data() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.set_column(&0, &vec![0, 0, 0]).unwrap().data,
                    vec![0, 1, 2, 0, 11, 12, 0, 21, 22]
                );
            }
        }

        mod set_columns {
            use super::*;

            #[test]
            fn empty_range() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.set_columns(&(0..0), &Vec::<Vec<u8>>::new()).unwrap().data,
                    vec![0, 1, 2, 10, 11, 12, 20, 21, 22]
                );
            }

            #[test]
            fn invalid_columns() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.set_columns(&(5..6), &Vec::<Vec<u8>>::new()), Err(6));
            }

            #[test]
            fn inconsistant_length() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.set_columns(&(0..1), &vec![vec![7]]), Err(1));
            }

            #[test]
            fn correct_data() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.set_columns(&(0..2), &vec![vec![1, 0, 0], vec![0, 1, 0]])
                        .unwrap()
                        .data,
                    vec![1, 0, 2, 0, 1, 12, 0, 0, 22]
                );
            }
        }
    }

    mod swappers {
        use super::*;

        mod swap_elements {
            use super::*;

            #[test]
            fn invalid_row() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.swap_elements((&4, &1), (&1, &2)), Err(4));
                assert_eq!(m.swap_elements((&0, &2), (&3, &0)), Err(3));
            }

            #[test]
            fn invalid_column() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.swap_elements((&0, &8), (&2, &1)), Err(8));
                assert_eq!(m.swap_elements((&1, &1), (&0, &6)), Err(6));
            }

            #[test]
            fn correct_data() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.swap_elements((&0, &0), (&1, &1)).unwrap().data,
                    vec![11, 1, 2, 10, 0, 12, 20, 21, 22]
                )
            }
        }

        mod swap_rows {
            use super::*;

            #[test]
            fn invalid_row() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.swap_rows(&7, &2), Err(7));
                assert_eq!(m.swap_rows(&0, &3), Err(3));
            }

            #[test]
            fn correct_data() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.swap_rows(&0, &2).unwrap().data,
                    vec![20, 21, 22, 10, 11, 12, 0, 1, 2]
                );
            }
        }

        mod swap_columns {
            use super::*;

            #[test]
            fn invalid_column() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(m.swap_columns(&6, &2), Err(6));
                assert_eq!(m.swap_columns(&1, &25), Err(25));
            }

            #[test]
            fn correct_data() {
                let mut m =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    m.swap_columns(&0, &2).unwrap().data,
                    vec![2, 1, 0, 12, 11, 10, 22, 21, 20]
                );
            }
        }
    }

    mod operations {
        use super::*;

        mod scale {
            use super::*;

            #[test]
            fn correct_data() {
                let mut m: Matrix<u8> = Matrix::new_identity(&3).unwrap();

                assert_eq!(m.scale(&2).data, vec![2, 0, 0, 0, 2, 0, 0, 0, 2]);
            }
        }

        mod scale_row {
            use super::*;

            #[test]
            fn invalid_row() {
                let mut m: Matrix<u8> = Matrix::new_identity(&3).unwrap();

                assert_eq!(m.scale_row(&4, &8), Err(4));
            }

            #[test]
            fn correct_data() {
                let mut m: Matrix<u8> = Matrix::new_identity(&3).unwrap();

                assert_eq!(
                    m.scale_row(&1, &5).unwrap().data,
                    vec![1, 0, 0, 0, 5, 0, 0, 0, 1]
                );
            }
        }

        mod add_scaled_row {
            use super::*;

            #[test]
            fn invalid_row() {
                let mut m: Matrix<u8> = Matrix::new_identity(&3).unwrap();

                assert_eq!(m.add_scaled_row(&9, &1, &3), Err(9));
                assert_eq!(m.add_scaled_row(&2, &48, &(1 / 2)), Err(48));
            }

            #[test]
            fn correct_data() {
                let mut m: Matrix<u8> = Matrix::new_identity(&3).unwrap();

                assert_eq!(
                    m.add_scaled_row(&0, &2, &10).unwrap().data,
                    vec![1, 0, 0, 0, 1, 0, 10, 0, 1]
                );
            }
        }

        mod scale_column {
            use super::*;

            #[test]
            fn invalid_column() {
                let mut m: Matrix<u8> = Matrix::new_identity(&3).unwrap();

                assert_eq!(m.scale_column(&5, &3), Err(5));
            }

            #[test]
            fn correct_data() {
                let mut m: Matrix<u8> = Matrix::new_identity(&3).unwrap();

                assert_eq!(
                    m.scale_column(&1, &2).unwrap().data,
                    vec![1, 0, 0, 0, 2, 0, 0, 0, 1]
                );
            }
        }

        mod add_scaled_column {
            use super::*;

            #[test]
            fn invalid_column() {
                let mut m: Matrix<u8> = Matrix::new_identity(&3).unwrap();

                assert_eq!(m.add_scaled_column(&72, &2, &90), Err(72));
                assert_eq!(m.add_scaled_column(&2, &4, &1), Err(4));
            }

            #[test]
            fn correct_data() {
                let mut m: Matrix<u8> = Matrix::new_identity(&3).unwrap();

                assert_eq!(
                    m.add_scaled_column(&0, &2, &3).unwrap().data,
                    vec![1, 0, 3, 0, 1, 0, 0, 0, 1]
                );
            }
        }
    }

    mod derivers {
        use super::*;

        mod transpose {
            use super::*;

            #[test]
            fn correct_data() {
                let m: Matrix<u8> =
                    Matrix::new_with_data(&4, &2, &vec![00, 01, 10, 11, 20, 21, 30, 31]).unwrap();

                assert_eq!(m.transpose(), {
                    Matrix {
                        rows: 2,
                        columns: 4,
                        data: vec![00, 10, 20, 30, 01, 11, 21, 31],
                    }
                });
            }
        }

        mod minor {
            use super::*;

            #[test]
            fn non_square_matrix() {
                let m: Matrix<i8> = Matrix::new(&3, &4).unwrap();

                assert_eq!(m.minor((&0, &0)), Err((3, 4)));
            }

            #[test]
            fn invalid_position() {
                let m: Matrix<i8> = Matrix::new_identity(&4).unwrap();

                assert_eq!(m.minor((&4, &2)), Err((4, 2)));
                assert_eq!(m.minor((&3, &6)), Err((3, 6)));
            }

            #[test]
            fn correct_data() {
                let m: Matrix<i8> = Matrix::new_identity(&4).unwrap();

                assert_eq!(m.minor((&0, &0)).unwrap(), 1);
                assert_eq!(m.minor((&0, &1)).unwrap(), 0);
            }
        }

        mod minor_matrix {
            use super::*;

            #[test]
            fn non_square_matrix() {
                let m: Matrix<i8> = Matrix::new(&5, &2).unwrap();

                assert_eq!(m.minor_matrix(), Err((5, 2)));
            }

            #[test]
            fn correct_data() {
                let m: Matrix<i8> = Matrix::new_identity(&4).unwrap();

                assert_eq!(m.minor_matrix().unwrap().data, m.data);
            }
        }

        mod cofactor {
            use super::*;

            #[test]
            fn correct_data() {
                let m: Matrix<i8> = Matrix::new_from_data(&vec![
                    vec![1, 1, 1, 1],
                    vec![1, 1, 1, 1],
                    vec![1, 1, 1, 1],
                ])
                .unwrap();

                assert_eq!(
                    m.cofactor().data,
                    vec![1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1]
                );
            }
        }

        mod adjoint {
            use super::*;

            #[test]
            fn non_square_matrix() {
                let m: Matrix<i8> = Matrix::new(&3, &5).unwrap();

                assert_eq!(m.adjoint(), Err((m.rows, m.columns)));
            }

            #[test]
            fn correct_data() {
                let m: Matrix<i8> = Matrix::new_identity(&5).unwrap();

                assert_eq!(m.adjoint().unwrap().data, m.data);
            }
        }

        mod determinant {
            use super::*;

            #[test]
            fn non_square_matrix() {
                let m: Matrix<i8> = Matrix::new(&3, &4).unwrap();

                assert_eq!(m.determinant(), Err((3, 4)));
            }

            #[test]
            fn correct_data() {
                let m: Matrix<i8> = Matrix::new_identity(&4).unwrap();

                assert_eq!(m.determinant().unwrap(), 1);
            }
        }

        mod inverse {
            use super::*;

            #[test]
            fn non_square_matrix() {
                let m: Matrix<i8> = Matrix::new(&3, &7).unwrap();

                assert_eq!(m.inverse(), Err((3, 7)));
            }

            #[test]
            fn correct_data() {
                let m: Matrix<i8> = Matrix::new_identity(&2).unwrap();

                assert_eq!(m.inverse().unwrap().data, m.data);
            }
        }
    }

    mod traits {
        use super::*;

        mod add {
            use super::*;

            #[test]
            fn different_size() {
                let m1: Matrix<i8> = Matrix::new(&3, &4).unwrap();
                let m2: Matrix<i8> = Matrix::new(&5, &4).unwrap();
                let m3: Matrix<i8> = Matrix::new(&3, &4).unwrap();
                let m4: Matrix<i8> = Matrix::new(&3, &6).unwrap();

                assert_eq!(m1 + m2, Err((3, 5)));
                assert_eq!(m3 + m4, Err((4, 6)));
            }

            #[test]
            fn correct_data() {
                let m1: Matrix<i8> = Matrix::new_identity(&3).unwrap();
                let m2: Matrix<i8> = Matrix::new_identity(&3).unwrap();

                assert_eq!((m1 + m2).unwrap().data, vec![2, 0, 0, 0, 2, 0, 0, 0, 2]);
            }
        }

        mod sub {
            use super::*;

            #[test]
            fn different_size() {
                let m1: Matrix<i8> = Matrix::new(&3, &3).unwrap();
                let m2: Matrix<i8> = Matrix::new(&2, &3).unwrap();
                let m3: Matrix<i8> = Matrix::new(&3, &3).unwrap();
                let m4: Matrix<i8> = Matrix::new(&3, &4).unwrap();

                assert_eq!(m1 - m2, Err((3, 2)));
                assert_eq!(m3 - m4, Err((3, 4)));
            }

            #[test]
            fn correct_data() {
                let m1: Matrix<i8> = Matrix::new_identity(&3).unwrap();
                let m2: Matrix<i8> = Matrix::new_identity(&3).unwrap();

                assert_eq!((m1 - m2).unwrap().data, vec![0, 0, 0, 0, 0, 0, 0, 0, 0]);
            }
        }

        mod mul {
            use super::*;

            #[test]
            fn invalid_size() {
                let m1: Matrix<i8> = Matrix::new(&3, &4).unwrap();
                let m2: Matrix<i8> = Matrix::new(&3, &3).unwrap();

                assert_eq!(m1 * m2, Err((4, 3)));
            }

            #[test]
            fn correct_data() {
                let m1 = Matrix::new_from_data(&vec![vec![0, 1], vec![2, 3]]).unwrap();
                let m2 = Matrix::new_from_data(&vec![vec![3, 2], vec![1, 0]]).unwrap();

                assert_eq!((m1 * m2).unwrap().data, vec![1, 0, 9, 4]);
            }

            #[test]
            fn identity() {
                let m1 = Matrix::new_identity(&3).unwrap();
                let m2 =
                    Matrix::new_from_data(&vec![vec![0, 1, 2], vec![10, 11, 12], vec![20, 21, 22]])
                        .unwrap();

                assert_eq!(
                    (m1 * m2).unwrap().data,
                    vec![0, 1, 2, 10, 11, 12, 20, 21, 22]
                );
            }
        }
    }
}
