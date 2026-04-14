use num_traits::Zero;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Mul};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    InvalidShape,
    ShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    NonContiguous,
    MatMulShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    InvalidAxes,
}

fn numel(shape: &[usize]) -> Result<usize, TensorError> {
    if shape.iter().any(|&d| d == 0) {
        return Err(TensorError::InvalidShape);
    }
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d).ok_or(TensorError::InvalidShape))
}

fn row_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut stride = 1isize;
    for (i, &dim) in shape.iter().enumerate().rev() {
        strides[i] = stride;
        stride *= dim as isize;
    }
    strides
}

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    data: Arc<[T]>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

impl<T> Tensor<T> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        // safe: all tensors we construct validate shape
        self.shape.iter().product()
    }

    pub fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.strides == row_major_strides(&self.shape)
    }

    pub fn as_slice(&self) -> Option<&[T]> {
        if !self.is_contiguous() {
            return None;
        }
        Some(&self.data)
    }

    pub fn from_vec(shape: &[usize], data: Vec<T>) -> Result<Self, TensorError> {
        let n = numel(shape)?;
        if data.len() != n {
            return Err(TensorError::InvalidShape);
        }
        Ok(Self {
            data: Arc::from(data),
            shape: shape.to_vec(),
            strides: row_major_strides(shape),
            offset: 0,
        })
    }

    /// Create a view with the same backing storage but new metadata.
    fn view_with(&self, shape: Vec<usize>, strides: Vec<isize>, offset: usize) -> Self {
        Self {
            data: Arc::clone(&self.data),
            shape,
            strides,
            offset,
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        if !self.is_contiguous() {
            return Err(TensorError::NonContiguous);
        }
        let old_n = numel(&self.shape)?;
        let new_n = numel(new_shape)?;
        if old_n != new_n {
            return Err(TensorError::InvalidShape);
        }
        Ok(self.view_with(
            new_shape.to_vec(),
            row_major_strides(new_shape),
            0,
        ))
    }

    /// Permute axes (general transpose). For example, for 2D transpose use axes [1, 0].
    pub fn permute_axes(&self, axes: &[usize]) -> Result<Self, TensorError> {
        if axes.len() != self.rank() {
            return Err(TensorError::InvalidAxes);
        }
        let mut seen = vec![false; self.rank()];
        for &a in axes {
            if a >= self.rank() || seen[a] {
                return Err(TensorError::InvalidAxes);
            }
            seen[a] = true;
        }

        let shape = axes.iter().map(|&a| self.shape[a]).collect::<Vec<_>>();
        let strides = axes.iter().map(|&a| self.strides[a]).collect::<Vec<_>>();
        Ok(self.view_with(shape, strides, self.offset))
    }
}

impl<T> Tensor<T>
where
    T: Copy,
{
    /// Materialize the tensor into a contiguous Vec in row-major order.
    pub fn to_vec_contiguous(&self) -> Vec<T> {
        if let Some(slice) = self.as_slice() {
            return slice.to_vec();
        }

        let rank = self.rank();
        if rank == 0 {
            return vec![];
        }

        let mut out = Vec::with_capacity(self.numel());
        let mut idx = vec![0usize; rank];

        loop {
            let mut off: isize = self.offset as isize;
            for d in 0..rank {
                off += (idx[d] as isize) * self.strides[d];
            }
            out.push(self.data[off as usize]);

            // increment odometer
            let mut dim = rank;
            while dim > 0 {
                dim -= 1;
                idx[dim] += 1;
                if idx[dim] < self.shape[dim] {
                    break;
                }
                idx[dim] = 0;
            }
            if dim == 0 && idx[0] == 0 {
                break;
            }
        }
        out
    }
}

impl<T> Tensor<T>
where
    T: Copy + Send + Sync + Add<Output = T>,
{
    pub fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape != rhs.shape {
            return Err(TensorError::ShapeMismatch {
                left: self.shape.clone(),
                right: rhs.shape.clone(),
            });
        }

        // Fast path: both contiguous
        if let (Some(a), Some(b)) = (self.as_slice(), rhs.as_slice()) {
            let out: Vec<T> = a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x + y).collect();
            return Tensor::from_vec(&self.shape, out);
        }

        // Slow path: materialize both
        let a = self.to_vec_contiguous();
        let b = rhs.to_vec_contiguous();
        let out: Vec<T> = a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x + y).collect();
        Tensor::from_vec(&self.shape, out)
    }
}

impl<T> Tensor<T>
where
    T: Copy + Send + Sync + Zero + AddAssign + Mul<Output = T>,
{
    /// Matrix multiplication for 2D tensors: (m×k) @ (k×n) -> (m×n)
    ///
    /// This is a cache-friendly row-parallel implementation (Rayon over output rows).
    pub fn matmul(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.rank() != 2 || rhs.rank() != 2 {
            return Err(TensorError::MatMulShapeMismatch {
                left: self.shape.clone(),
                right: rhs.shape.clone(),
            });
        }
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (rhs.shape[0], rhs.shape[1]);
        if k1 != k2 {
            return Err(TensorError::MatMulShapeMismatch {
                left: self.shape.clone(),
                right: rhs.shape.clone(),
            });
        }

        let a = self.to_vec_contiguous();
        let b = rhs.to_vec_contiguous();

        let mut out = vec![T::zero(); m * n];

        // Block sizes (simple heuristics). These can be tuned later.
        let bk = 32usize;
        let bj = 64usize;

        out.par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, out_row)| {
                let a_row = &a[i * k1..(i + 1) * k1];

                // Block over k and n for better cache locality.
                let mut kk = 0usize;
                while kk < k1 {
                    let kend = (kk + bk).min(k1);
                    let mut jj = 0usize;
                    while jj < n {
                        let jend = (jj + bj).min(n);
                        for k in kk..kend {
                            let aval = a_row[k];
                            let b_row = &b[k * n..(k + 1) * n];
                            for j in jj..jend {
                                out_row[j] += aval * b_row[j];
                            }
                        }
                        jj = jend;
                    }
                    kk = kend;
                }
            });

        Tensor::from_vec(&[m, n], out)
    }
}

impl<T> Tensor<T>
where
    T: Copy + Zero,
{
    pub fn zeros(shape: &[usize]) -> Result<Self, TensorError> {
        let n = numel(shape)?;
        Tensor::from_vec(shape, vec![T::zero(); n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_and_contiguous() {
        let t = Tensor::from_vec(&[2, 3], vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.is_contiguous());
        assert_eq!(t.as_slice().unwrap().len(), 6);
    }

    #[test]
    fn reshape_contiguous_ok() {
        let t = Tensor::from_vec(&[2, 3], vec![1u32, 2, 3, 4, 5, 6]).unwrap();
        let r = t.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert!(r.is_contiguous());
        assert_eq!(r.to_vec_contiguous(), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn permute_axes_transpose_2d() {
        let t = Tensor::from_vec(&[2, 3], vec![1u32, 2, 3, 4, 5, 6]).unwrap();
        let tt = t.permute_axes(&[1, 0]).unwrap();
        assert_eq!(tt.shape(), &[3, 2]);
        // materialized transpose:
        assert_eq!(tt.to_vec_contiguous(), vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn add_elementwise() {
        let a = Tensor::from_vec(&[2, 2], vec![1i32, 2, 3, 4]).unwrap();
        let b = Tensor::from_vec(&[2, 2], vec![10i32, 20, 30, 40]).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.to_vec_contiguous(), vec![11, 22, 33, 44]);
    }

    #[test]
    fn matmul_works() {
        // (2x3) @ (3x2) = (2x2)
        let a = Tensor::from_vec(&[2, 3], vec![1i32, 2, 3, 4, 5, 6]).unwrap();
        let b = Tensor::from_vec(&[3, 2], vec![7i32, 8, 9, 10, 11, 12]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]]
        // = [[58,64],[139,154]]
        assert_eq!(c.to_vec_contiguous(), vec![58, 64, 139, 154]);
    }
}
