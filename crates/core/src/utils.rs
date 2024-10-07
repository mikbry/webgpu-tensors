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
}