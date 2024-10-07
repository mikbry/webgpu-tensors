#[macro_export]
macro_rules! n_vec {
    ($arr:expr) => {{                                                                 
        fn get_shape_and_flatten<T: Copy + Into<f32>>(arr: &[T]) -> (   
Vec<f32>, Vec<usize>) {                                                                           
            (arr.iter().map(|&x| x.into()).collect(), vec![arr.len()])                
        }                                                                             
                                                                                      
        fn recurse<T: Copy + Into<f32>>(arr: &[T]) -> (Vec<usize>, Vec<f32>) {        
            if let Some(first) = arr.get(0) {                                         
                if let Some(nested) = (first as *const T).cast::<[T]>().as_ref() {    
                    let (mut shape, flattened) = recurse(nested);                     
                    shape.insert(0, arr.len());                                       
                    (shape, flattened.into_iter().cycle().take(flattened.len() *      
arr.len()).collect())                                                                 
                } else {                                                              
                    get_shape_and_flatten(arr)                                        
                }                                                                     
            } else {                                                                  
                (vec![], vec![])                                                      
            }                                                                         
        }                                                                             
                                                                                      
        recurse(&$arr)                                                                
    }};                                                                               
}use std::iter::FromIterator;

#[macro_export]
macro_rules! to_flat_vec_and_shape {
    ($arr:expr) => {{
        fn get_shape_and_flatten<T: Copy + Into<f32>>(arr: &[T]) -> (Vec<usize>, Vec<f32>) {
            (vec![arr.len()], arr.iter().map(|&x| x.into()).collect())
        }

        fn process_array<T: Copy + Into<f32>>(arr: &[T]) -> (Vec<usize>, Vec<f32>) {
            match arr {
                [] => (vec![], vec![]),
                [first, ..] => {
                    if let Some(nested) = first.as_ref().and_then(|x| x.as_slice()) {
                        let (inner_shape, mut flattened) = process_array(nested);
                        let mut shape = vec![arr.len()];
                        shape.extend(inner_shape);
                        for item in &arr[1..] {
                            if let Some(nested) = item.as_ref().and_then(|x| x.as_slice()) {
                                flattened.extend(nested.iter().map(|&x| x.into()));
                            }
                        }
                        (shape, flattened)
                    } else {
                        get_shape_and_flatten(arr)
                    }
                }
            }
        }

        process_array(&$arr)
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_to_flat_vec_and_shape() {
        let arr1 = [1, 2, 3, 4];
        assert_eq!(to_flat_vec_and_shape!(arr1), (vec![4], vec![1.0, 2.0, 3.0, 4.0]));

        let arr2 = [[1, 2], [3, 4]];
        assert_eq!(to_flat_vec_and_shape!(arr2), (vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));

        let arr3 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]];
        assert_eq!(to_flat_vec_and_shape!(arr3), (vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
    }
}
