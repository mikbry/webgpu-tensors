
#[macro_export]
macro_rules! to_flat_vec_and_shape {
    ($arr:expr) => {{
        let mut shape = Vec::new();
        let mut flat_vec = Vec::new();

        fn process_array<T: Copy + Into<f32>>(arr: &[T], shape: &mut Vec<usize>, flat_vec: &mut Vec<f32>) {
            shape.push(arr.len());
            flat_vec.extend(arr.iter().map(|&x| x.into()));
        }

        fn process_nested_array<T: Copy + Into<f32>>(arr: &[T], shape: &mut Vec<usize>, flat_vec: &mut Vec<f32>) {
            if arr.is_empty() {
                return;
            }
            shape.push(arr.len());
            
            let first = &arr[0];
            if let Some(nested) = (first as &dyn std::any::Any).downcast_ref::<[T]>() {
                process_nested_array(nested, shape, flat_vec);
                for item in arr.iter().skip(1) {
                    if let Some(nested) = (item as &dyn std::any::Any).downcast_ref::<[T]>() {
                        process_nested_array(nested, shape, flat_vec);
                    }
                }
            } else {
                process_array(arr, shape, flat_vec);
            }
        }

        process_nested_array(&[$arr], &mut shape, &mut flat_vec);

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

        let arr4 = [[1.0, 2.0], [3, 4]];
        assert_eq!(to_flat_vec_and_shape!(arr4), (vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    }
}
