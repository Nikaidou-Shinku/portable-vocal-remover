pub fn hann_window(length: usize) -> Vec<f32> {
  if length == 0 {
    return Vec::new();
  }

  if length == 1 {
    return vec![1.];
  }

  let half_length = (length + 1) / 2;

  let factor = (std::f32::consts::PI * 2.) / (length - 1) as f32;

  let mut res = vec![0.0; length];

  for i in 0..half_length {
    let cur = 0.5 - 0.5 * (i as f32 * factor).cos();
    res[i] = cur;
    res[length - i - 1] = cur;
  }

  res
}
