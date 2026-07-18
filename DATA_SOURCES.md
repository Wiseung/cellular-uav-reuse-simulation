# 数据源说明

*项目输入数据的来源、处理方式、许可边界和复现限制。*

---

## 📍 已随仓库提供的输入

| 文件 | 类型 | 来源或性质 | 用途 |
| --- | --- | --- | --- |
| `real_site_layout_knoxville_tn.csv` | CSV | 从 ArcGIS Hub 的公开美国蜂窝塔数据中提取的 Knoxville, Tennessee 局部站点簇 | 动态站点布局和站点级参数输入 |
| `knoxville_site_layout_buildings.geojson` | GeoJSON | 通过 Overpass API 获取的 OpenStreetMap 建筑轮廓 | 动态链路的建筑遮挡、LOS/NLOS 和额外损耗 |
| `demo_site_layout.csv` | CSV | 项目内构造的规则化演示布局 | 测试和快速示例 |
| `custom_panel_pattern.csv` | CSV | 项目内参考天线水平/垂直衰减曲线 | 方向图格式测试和 MSI 文件生成源 |
| `reference_sector_panel_pattern.msi` | MSI | 由 `custom_panel_pattern.csv` 插值生成的 MSI Planet 风格参考文件 | 默认扇区天线方向图 |
| `default_parameter_profile.json` | JSON | 项目运行配置 | 波束、负载、切换、协调和路径配置 |
| `building_material_loss_profile.json` | JSON | 项目建模假设 | 建筑材料与穿透损耗映射 |

## 🗼 Knoxville 站点布局

`real_site_layout_knoxville_tn.csv` 来源于 ArcGIS Hub 数据集 **Cellular Towers in the United States**。处理步骤如下：

1. 获取带几何和位置属性的公开站点记录。
2. 在公开点集中搜索 Knoxville 中心点附近的局部密集站点簇。
3. 选择中心点及其附近的 21 个站点。
4. 将经纬度转换为以中心站点为原点的局部米制坐标 `x_m`、`y_m`。

该文件只用于当前局部动态实验，不代表全美蜂窝塔数据，也不包含完整的运营商无线参数。文件中的 `objectid`、城市、州、纬度、经度和距离字段保留了部分公开记录信息。

公开入口：

- 数据集 API：<https://hub.arcgis.com/api/v3/datasets/15dabb4108254481b591018be2598f3c_0>
- CSV 下载接口：<https://hub.arcgis.com/api/v3/datasets/15dabb4108254481b591018be2598f3c_0/downloads/data?format=csv&spatialRefId=4326>

## 🏢 建筑轮廓

`knoxville_site_layout_buildings.geojson` 通过公开 Overpass API 下载 OpenStreetMap 的 `way["building"]` 要素生成。处理步骤如下：

1. 根据 Knoxville 站点布局计算覆盖范围，并向外扩展 `0.003°`。
2. 将范围切分为 `0.04°` 瓦片，并设置 `0.001°` 重叠区域。
3. 分瓦片查询建筑要素，对临时 `429` 和 `5xx` 响应进行重试。
4. 按 `osm_way_id` 去重并合并为 GeoJSON `FeatureCollection`。
5. 仿真运行时将多边形重新投影到站点布局使用的局部米制坐标。

该文件是二维建筑平面样本，不是包含楼层、纹理和室内结构的完整三维城市模型。建筑高度和材料字段缺失时，仿真使用配置中的默认高度和材料损耗参数；存在相应属性时才应用材料覆盖。

- Overpass API：<https://overpass-api.de/api/interpreter>
- OpenStreetMap 版权与许可说明：<https://www.openstreetmap.org/copyright>

使用或重新下载 OpenStreetMap 数据时，应遵守当前 ODbL 要求、署名要求和 Overpass 服务限制。

## 📡 天线方向图

`custom_panel_pattern.csv` 是项目内的参考水平/垂直衰减曲线，不是制造商测量数据。`reference_sector_panel_pattern.msi` 的生成过程为：

1. 读取 CSV 中的水平和垂直衰减点。
2. 将曲线插值到密集角度网格。
3. 写入包含 `NAME`、`MAKE`、`FREQUENCY`、`H_WIDTH`、`V_WIDTH`、`GAIN` 和 `TILT` 等字段的 MSI Planet 风格文本。

因此，MSI 文件用于格式兼容和可重复仿真，不应解释为真实厂商天线的认证方向图。若有授权的厂商 `.msi` 或 `.pln` 文件，可以通过配置替换默认方向图。

## ⚙️ 参数和材料配置

`default_parameter_profile.json`、`building_material_loss_profile.json` 和 `demo_site_layout.csv` 是项目内配置或演示输入，不是外部测量数据库：

- 参数 Profile 将波束、负载、切换、协调调度和路径覆盖设置外置化。
- 材料 Profile 通过关键词、穿透损耗倍率和入口损耗表达建模假设。
- Demo 布局用于测试，不表示真实地理位置。

这些文件适合用于复现实验流程；如果用于工程决策，应替换为经过授权、校准和版本记录的现场数据。

## 🌐 可选外部数据流程

仓库不包含原始公开数据下载包。下列工具在显式运行时处理用户提供的文件或访问外部服务：

| 工具 | 输入 | 外部来源 |
| --- | --- | --- |
| `tools/extract_arcgis_site_cluster.py` | ArcGIS 站点数据或其 API 响应 | ArcGIS Hub |
| `tools/download_osm_buildings.py` | 站点布局 CSV | OpenStreetMap Overpass |
| `tools/prepare_enhanced_site_layout.py` | OpenCelliD 或 CellMapper CSV | 用户提供的公开蜂窝数据导出 |
| `tools/prepare_overture_3dep_buildings.py` | Overture 建筑 GeoJSON | Overture Maps；可选 USGS 3DEP 高程服务 |
| `tools/run_public_data_pipeline.py` | 原始小区 CSV、建筑 GeoJSON | 以上准备流程的组合 |

可选数据的许可证、更新时间、字段含义、空间精度和服务配额由各数据提供方决定。运行前应保存原始数据版本、下载日期、查询范围和许可信息，以便复现实验。

## 🔁 复现建议

1. 记录每个外部文件的下载地址、时间、版本或校验值。
2. 保留原始文件在仓库外部，不要把下载包、密钥或临时结果提交到 Git。
3. 使用准备工具生成标准化 CSV/GeoJSON，再将生成路径传给仿真程序。
4. 将 `default_parameter_profile.json` 复制为实验专用 Profile，并记录修改项。
5. 对比不同数据源时固定仿真随机种子、中心坐标、筛选半径和站点数量。

## ⚠️ 数据限制

- Knoxville 站点布局是局部样本，不能推断全国网络覆盖或实际运营商配置。
- OpenStreetMap 建筑轮廓主要提供平面几何，建筑高度和材料覆盖可能不完整。
- 天线方向图和材料损耗 Profile 是参考建模输入，不替代实测校准。
- 公开数据的字段、许可和 API 行为可能变化；重新生成的数据不一定与仓库内样本逐字节一致。
