fn main() -> anyhow::Result<()> {
    let model_onnx_path = std::path::Path::new("./python/model.onnx");
    let graph = onnx_ir::parse_onnx(model_onnx_path);

    use tract_onnx::tract_core::framework::Framework;
    let model = tract_onnx::onnx()
        .model_for_path(model_onnx_path)?
        .into_compact()?
        .into_runnable()?;
    use tract_onnx::prelude::*;

    let model_input = model.model().outlet_fact(model.model().inputs[0])?;
    println!("{model_input:?}");

    let input: Tensor = tract_ndarray::arr2(&[[1_f32; 64]]).into();

    let input = tvec!(input.into());

    let result = model.run(input).unwrap();
    println!("{result:?}");

    Ok(())
}
