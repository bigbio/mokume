# Community

## Citation

If you use mokume in your research, please cite:

> Zheng P, Audain E, Webel H, Dai C, Klein J, Hitz MP, Sachsenberg T, Bai M, Perez-Riverol Y. Ibaqpy: A scalable Python package for baseline quantification in proteomics leveraging SDRF metadata. *J Proteomics*. 2025;317:105440. doi: [10.1016/j.jprot.2025.105440](https://doi.org/10.1016/j.jprot.2025.105440).

> Wang H, Dai C, Pfeuffer J, Sachsenberg T, Sanchez A, Bai M, Perez-Riverol Y. Tissue-based absolute quantification using large-scale TMT and LFQ experiments. *Proteomics*. 2023;23(20):e2300188. doi: [10.1002/pmic.202300188](https://doi.org/10.1002/pmic.202300188).

## Credits

mokume is developed by the [BigBio](https://github.com/bigbio) team:

- [Julianus Pfeuffer](https://github.com/jpfeuffer)
- [Yasset Perez-Riverol](https://github.com/ypriverol)
- [Hong Wang](https://github.com/WangHong007)
- [Ping Zheng](https://github.com/zprobot)
- [Joshua Klein](https://github.com/mobiusklein)
- [Enrique Audain](https://github.com/enriquea)

## Contributing

We welcome contributions! To get started:

1. Fork the [repository](https://github.com/bigbio/mokume)
2. Create a feature branch
3. Make your changes
4. Run the test suite for each implementation or periphery area you changed
5. Submit a pull request

### Development Setup

mokume contains a leading Rust compute kernel (`rust/crates/`), its wheel
periphery (`rust/python/mokume/`), and a separately maintained pure-Python
implementation (`python/mokume/`). New native computation lands in Rust first;
see [Maintenance scope](maintenance-scope.md) for when a matching pure-Python
change is expected.

The Rust toolchain is pinned to **1.100.0** via `rust-toolchain.toml` (with the `rustfmt` and `clippy` components), so `rustup` selects it automatically inside the checkout.

```bash
git clone https://github.com/bigbio/mokume
cd mokume/rust

# Rust kernel: build, test, format, lint (24 threads per the project convention)
cargo build --jobs 24
cargo test --jobs 24
cargo fmt --all
cargo clippy --jobs 24 --all-targets

# Python wheel: build the mokume._mokume extension into a dev environment,
# then run the periphery test suite
pip install maturin
maturin develop --extras all
pytest
```

`maturin develop` compiles the PyO3 binding crate (`crates/mokume-py`) into the in-process `mokume._mokume` extension and installs the wheel editable; the periphery under `rust/python/mokume/commands/` is plain Python and needs no build step. After changing kernel code, re-run `maturin develop` so the Python wrappers pick up the new extension.

For changes to the separately maintained pure-Python implementation:

```bash
cd ../python
pip install -e ".[all]"
pytest
```

## License

mokume is released under the [MIT License](https://github.com/bigbio/mokume/blob/main/LICENSE).

## Links

- **GitHub**: [github.com/bigbio/mokume](https://github.com/bigbio/mokume)
- **PyPI**: [pypi.org/project/mokume](https://pypi.org/project/mokume)
- **Issues**: [github.com/bigbio/mokume/issues](https://github.com/bigbio/mokume/issues)
- **quantms Ecosystem**: [quantms.org](https://quantms.org)
