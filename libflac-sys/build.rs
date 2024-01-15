fn main() {
  let mut cmake = cmake::Config::new("src/libflac");

  cmake
    .define("BUILD_CXXLIBS", "OFF")
    .define("BUILD_DOCS", "OFF")
    .define("BUILD_EXAMPLES", "OFF")
    .define("BUILD_PROGRAMS", "OFF")
    .define("INSTALL_CMAKE_CONFIG_MODULE", "OFF")
    .define("INSTALL_MANPAGES", "OFF")
    .define("INSTALL_PKGCONFIG_MODULES", "OFF")
    .define("WITH_OGG", "OFF");

  let install_dir = cmake.build();

  let includedir = install_dir.join("include");
  let libdir = install_dir.join("lib");
  println!(
    "cargo:rustc-link-search=native={}",
    libdir.to_str().unwrap()
  );
  // TODO: check this
  println!("cargo:rustc-link-lib=static=flac");
  println!("cargo:root={}", install_dir.to_str().unwrap());
  println!("cargo:include={}", includedir.to_str().unwrap());
}
