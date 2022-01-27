use std::ops::{Mul, Add, Sub, SubAssign, Div};
use std::cmp::PartialOrd;
use std::ops::AddAssign;
use std::fmt::{Formatter, Display, Write, Debug};
use rand::distributions::Standard;
use rand::{thread_rng, Rng};
use rand::prelude::Distribution;

pub trait MatrixCell<T>: Default + Clone + Copy + Sub<Output = T> + Mul<Output = T> + AddAssign + SubAssign + Display + Add<Output = T> + Div<Output = T> + PartialOrd + Debug {}

impl MatrixCell<f32> for f32 {}
impl MatrixCell<f64> for f64 {}
impl MatrixCell<u32> for u32 {}
impl MatrixCell<i32> for i32 {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix<T, const HEIGHT: usize, const WIDTH: usize>
where T: MatrixCell<T>
{
    data: [[T; WIDTH]; HEIGHT]
}

impl<T, const HEIGHT: usize, const WIDTH: usize> Matrix<T, HEIGHT, WIDTH>
where T: MatrixCell<T>
{
    pub fn new() -> Self {
        Matrix::fill_with(T::default())
    }

    pub fn fill_with(val: T) -> Self {
        Self {
            data: [[val; WIDTH]; HEIGHT]
        }
    }

    pub fn from_arrays(data: [[T; WIDTH]; HEIGHT]) -> Self {
        Self {
            data
        }
    }

    pub fn apply(mut self, func: fn(T) -> T) -> Self {
        self.iter_mut().for_each(|t| *t = func(*t));
        self
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= HEIGHT || col >= WIDTH {
            None
        } else {
            Some(&self.data[row][col])
        }
    }


    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row >= HEIGHT || col >= WIDTH {
            None
        } else {
            Some(&mut self.data[row][col])
        }
    }

    pub fn set(&mut self, row: usize, col: usize, val: T) {
        self.data[row][col] = val;
    }

    pub fn transpose(&self) -> Matrix<T, WIDTH, HEIGHT> {
        let mut mat = Matrix::<T, WIDTH, HEIGHT>::new();

        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                mat.set(col, row, *self.get(row, col).unwrap());
            }
        }

        mat
    }

    pub fn hadamard(mut self, other: Self) -> Self {
        self.iter_mut().zip(other.iter())
            .for_each(|(x, y)| {
                *x = *x * *y;
            });

        self
    }

    pub fn max(&self) -> T {
        *self.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    }

    pub fn max_index(&self) -> usize {
        self.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().flatten()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut().flatten()
    }

    pub fn rows(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..HEIGHT).map(|i| self.data[i].iter())
    }

    pub fn cols(&self) -> impl Iterator<Item = impl Iterator<Item = &T>> {
        (0..WIDTH).map(|col| self.data.iter().flatten().skip(col).step_by(HEIGHT))
    }
}

impl<T, const HEIGHT: usize> Matrix<T, HEIGHT, 1> where T: MatrixCell<T> {
    pub fn outer<const OTHER_HEIGHT: usize>(&self, other: Matrix<T, OTHER_HEIGHT, 1>) -> Matrix<T, HEIGHT, OTHER_HEIGHT> {
        let mut m = Matrix::<T, HEIGHT, OTHER_HEIGHT>::new();

        for row in 0..HEIGHT {
            for col in 0..OTHER_HEIGHT {
                m.set(row, col, *self.get(row, 0).unwrap() * *other.get(col, 0).unwrap())
            }
        }

        m
    }
}

impl<T, const WIDTH: usize, const HEIGHT: usize, const OTHER_WIDTH: usize> std::ops::Mul<Matrix<T, WIDTH, OTHER_WIDTH>> for Matrix<T, HEIGHT, WIDTH>
where T: MatrixCell<T>
{
    type Output = Matrix<T, HEIGHT, OTHER_WIDTH>;

    fn mul(self, rhs: Matrix<T, WIDTH, OTHER_WIDTH>) -> Self::Output {
        let mut out = Matrix::<T, HEIGHT, OTHER_WIDTH>::new();

        for i in 0..HEIGHT {
            for j in 0..OTHER_WIDTH {
                let mut sum = T::default();

                for k in 0..WIDTH {
                    sum += self.data[i][k]*rhs.data[k][j];
                }

                out.data[i][j] = sum;
            }
        }

        out
    }
}

impl<T, const WIDTH: usize, const HEIGHT: usize> Add<Matrix<T, HEIGHT, WIDTH>> for Matrix<T, HEIGHT, WIDTH>
where T: MatrixCell<T> {
    type Output = Matrix<T, HEIGHT, WIDTH>;

    fn add(self, rhs: Self) -> Self::Output {
        let mut mat = Matrix::<T, HEIGHT, WIDTH>::new();

        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                mat.set(row, col, *self.get(row, col).unwrap() + *rhs.get(row, col).unwrap())
            }
        }

        mat
    }
}

impl<T, const WIDTH: usize, const HEIGHT: usize> AddAssign<Matrix<T, HEIGHT, WIDTH>> for Matrix<T, HEIGHT, WIDTH>
where T: MatrixCell<T> {
    fn add_assign(&mut self, rhs: Self) {
        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                self.set(row, col, *self.get(row, col).unwrap() + *rhs.get(row, col).unwrap())
            }
        }
    }
}

impl<T, const WIDTH: usize, const HEIGHT: usize> SubAssign<Matrix<T, HEIGHT, WIDTH>> for Matrix<T, HEIGHT, WIDTH>
where T: MatrixCell<T> {
    fn sub_assign(&mut self, rhs: Self) {
        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                self.set(row, col, *self.get(row, col).unwrap() - *rhs.get(row, col).unwrap())
            }
        }
    }
}

impl<T, const WIDTH: usize, const HEIGHT: usize> Sub<Matrix<T, HEIGHT, WIDTH>> for Matrix<T, HEIGHT, WIDTH>
where T: MatrixCell<T> {
    type Output = Matrix<T, HEIGHT, WIDTH>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut mat = Matrix::<T, HEIGHT, WIDTH>::new();

        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                mat.set(row, col, *self.get(row, col).unwrap() - *rhs.get(row, col).unwrap())
            }
        }

        mat
    }
}

impl<T, const WIDTH: usize, const HEIGHT: usize> Mul<T> for Matrix<T, HEIGHT, WIDTH>
where T: MatrixCell<T> {
    type Output = Matrix<T, HEIGHT, WIDTH>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut mat = Matrix::<T, HEIGHT, WIDTH>::new();

        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                mat.set(row, col, *self.get(row, col).unwrap()*rhs)
            }
        }

        mat
    }
}

impl<T, const WIDTH: usize, const HEIGHT: usize> Div<T> for Matrix<T, HEIGHT, WIDTH>
where T: MatrixCell<T> {
    type Output = Matrix<T, HEIGHT, WIDTH>;

    fn div(self, rhs: T) -> Self::Output {
        let mut mat = Matrix::<T, HEIGHT, WIDTH>::new();

        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                mat.set(row, col, *self.get(row, col).unwrap()/rhs)
            }
        }

        mat
    }
}



impl<T, const WIDTH: usize, const HEIGHT: usize> Display for Matrix<T, HEIGHT, WIDTH>
where T: MatrixCell<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut rows = self.rows().peekable();

        while let Some(row) = rows.next() {
            for col in row {
                write!(f, "{} ", col)?;
            }

            if rows.peek().is_some() {
                f.write_char('\n')?;
            }
        }

        Ok(())
    }
}

impl<T, const WIDTH: usize, const HEIGHT: usize> Matrix<T, HEIGHT, WIDTH>
where T: MatrixCell<T>, Standard: Distribution<T> {
    pub fn random() -> Matrix<T, HEIGHT, WIDTH> {
        let mut mat = Matrix::<T, HEIGHT, WIDTH>::new();
        let mut rng = thread_rng();

        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                mat.set(row, col, rng.gen());
            }
        }

        mat
    }
}

// Test cases, because I might have screwed something up

#[test]
fn test_mult() {
    let a = Matrix::<f32,2,2>::from_arrays([
        [0.1, 0.4],
        [0.5, -0.3]
    ]);

    let b = Matrix::<f32,2,2>::from_arrays([
        [-2.3, 4.1],
        [2.0, 2.0]
    ]);

    let res = Matrix::<f32,2,2>::from_arrays([
        [0.57, 1.21],
        [-1.75, 1.4499999]
    ]);

    assert_eq!(res, a*b);
}

#[test]
fn test_apply() {
    let a = Matrix::<f32,2,2>::from_arrays([
        [0.1, 0.4],
        [0.5, -0.3]
    ]);

    let b = a.apply(|x| 2.0*x);

    let res = Matrix::<f32,2,2>::from_arrays([
        [0.2, 0.8],
        [1.0, -0.6]
    ]);

    assert_eq!(res, b);
}

#[test]
fn test_transpose() {
    let a = Matrix::<f32,3,2>::from_arrays([
        [0.1, -0.2],
        [4.3, 2.1],
        [-2.0, 2.0]
    ]);

    let res = Matrix::<f32,2,3>::from_arrays([
        [0.1, 4.3, -2.0],
        [-0.2, 2.1, 2.0]
    ]);

    assert_eq!(res, a.transpose());
}

#[test]
fn test_hadamard() {
    let a = Matrix::<f32,3,2>::from_arrays([
        [0.1, -0.2],
        [4.0, 2.1],
        [-2.0, 2.0]
    ]);
    let b = Matrix::<f32,3,2>::from_arrays([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]);

    let res = Matrix::<f32,3,2>::from_arrays([
        [0.1, -0.4],
        [12.0, 8.4],
        [-10.0, 12.0]
    ]);

    assert_eq!(res, a.hadamard(b));
}
