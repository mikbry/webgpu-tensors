
#[macro_export]
macro_rules! to_flat_vec_and_shape {
    ($arr:expr) => {{
        let mut shape = Vec::new();
        let mut flat_vec = Vec::new();

        // Helper function to process nested arrays
        fn process_array<T: Copy + Into<f32>>(arr: &[T], shape: &mut Vec<usize>, flat_vec: &mut Vec<f32>) {
            if arr.is_empty() {
                return;
            }
            shape.push(arr.len());
            if let Some(first) = arr.first() {
                if let Some(nested) = (first as &dyn std::any::Any).downcast_ref::<[T]>() {
                    process_array(nested, shape, flat_vec);
                    for item in arr.iter().skip(1) {
                        if let Some(nested) = (item as &dyn std::any::Any).downcast_ref::<[T]>() {
                            process_array(nested, shape, flat_vec);
                        }
                    }
                } else {
                    flat_vec.extend(arr.iter().map(|&x| x.into()));
                }
            }
        }

        process_array(&[$arr], &mut shape, &mut flat_vec);

        // Remove the outermost dimension if it's 1
        if shape[0] == 1 {
            shape.remove(0);
        }

        (shape, flat_vec)
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_to_flat_vec_and_shape() {
        let arr1 = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(to_flat_vec_and_shape!(arr1), (vec![4], vec![1.0, 2.0, 3.0, 4.0]));

        let arr2 = [[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(to_flat_vec_and_shape!(arr2), (vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));

        let arr3 = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        assert_eq!(to_flat_vec_and_shape!(arr3), (vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
    }
}
