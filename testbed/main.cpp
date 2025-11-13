#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace {

constexpr float camera_fov = 70.0f;
constexpr float camera_near_plane = 0.01f;
constexpr float camera_far_plane = 100.0f;

struct Matrix {
	float m[4][4];
};

struct Vector {
	float x, y, z;
};

struct Vertex {
	Vector position;
	// ПРИМЕЧАНИЕ: Здесь можно добавить дополнительные атрибуты
};

struct Cone {
	Vector position;
	float rotation;
	float rotation_speed;
	float scale;
};

// ПРИМЕЧАНИЕ: Эти переменные будут доступны шейдерам через push constant uniform
struct ShaderConstants {
	Matrix projection;
	Matrix transform;
	Vector color;
};

struct VulkanBuffer {
	VkBuffer buffer;
	VkDeviceMemory memory;
};

VkShaderModule vertex_shader_module;
VkShaderModule fragment_shader_module;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

// ПРИМЕЧАНИЕ: Объявляем буферы и другие переменные здесь
VulkanBuffer vertex_buffer;
VulkanBuffer index_buffer;
uint32_t index_count = 0;

constexpr int NUM_CONES = 3;
Cone cones[NUM_CONES] = {
	{{-3.5f, -1.0f, 6.0f}, 0.0f, 0.5f, 1.0f}, 
	{{0.0f, 0.5f, 6.0f}, 0.0f, 1.0f, 1.0f},
	{{3.5f, 2.0f, 6.0f}, 0.0f, 1.5f, 1.0f},
};

Vector fixed_color = {0.2f, 0.7f, 0.9f};

float rotation_speed_multiplier = 1.0f;

// Переключение проекций
bool use_perspective_projection = true;

// Анимация с паузой и реверсом
bool animation_paused = false;
float animation_direction = 1.0f;  // 1.0f = вперёд, -1.0f = назад

Matrix identity() {
	Matrix result{};

	result.m[0][0] = 1.0f;
	result.m[1][1] = 1.0f;
	result.m[2][2] = 1.0f;
	result.m[3][3] = 1.0f;
	
	return result;
}

Matrix projection(float fov, float aspect_ratio, float near, float far) {
	Matrix result{};

	const float radians = fov * M_PI / 180.0f;
	const float cot = 1.0f / tanf(radians / 2.0f);

	result.m[0][0] = cot / aspect_ratio;
	result.m[1][1] = cot;
	result.m[2][3] = 1.0f;

	result.m[2][2] = far / (far - near);
	result.m[3][2] = (-near * far) / (far - near);

	return result;
}

Matrix orthographic_projection(float left, float right, float bottom, float top, float near, float far) {
	Matrix result{};
	
	result.m[0][0] = 2.0f / (right - left);
	result.m[1][1] = 2.0f / (top - bottom);
	result.m[2][2] = 1.0f / (far - near);
	result.m[3][3] = 1.0f;
	
	result.m[3][0] = -(right + left) / (right - left);
	result.m[3][1] = -(top + bottom) / (top - bottom);
	result.m[3][2] = -near / (far - near);
	
	return result;
}

Matrix translation(Vector vector) {
	Matrix result = identity();

	result.m[3][0] = vector.x;
	result.m[3][1] = vector.y;
	result.m[3][2] = vector.z;

	return result;
}

Matrix scaling(float scale) {
	Matrix result = identity();
	
	result.m[0][0] = scale;
	result.m[1][1] = scale;
	result.m[2][2] = scale;
	
	return result;
}

Matrix rotation(Vector axis, float angle) {
	Matrix result{};

	float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

	axis.x /= length;
	axis.y /= length;
	axis.z /= length;

	float sina = sinf(angle);
	float cosa = cosf(angle);
	float cosv = 1.0f - cosa;

	// Формула вращения Родрига

	result.m[0][0] = (axis.x * axis.x * cosv) + cosa;
	result.m[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
	result.m[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);

	result.m[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
	result.m[1][1] = (axis.y * axis.y * cosv) + cosa;
	result.m[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);

	result.m[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
	result.m[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
	result.m[2][2] = (axis.z * axis.z * cosv) + cosa;

	result.m[3][3] = 1.0f; // делает матрицу корректной для использования в однородных координатах

	return result;
}

Matrix multiply(const Matrix& a, const Matrix& b) {
	Matrix result{};

	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			for (int k = 0; k < 4; k++) {
				result.m[j][i] += a.m[j][k] * b.m[k][i];
			}
		}
	}

	return result;
}

// ПРИМЕЧАНИЕ: Загружает байт-код шейдера из файла
// ПРИМЕЧАНИЕ: Ваши шейдеры компилируются через CMake, посмотрите этот код
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

VulkanBuffer createBuffer(size_t size, void *data, VkBufferUsageFlags usage) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;
	
	VulkanBuffer result{};

	{
		// ПРИМЕЧАНИЕ: Создаём буфер с указанным назначением и размером
		VkBufferCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};

		if (vkCreateBuffer(device, &info, nullptr, &result.buffer) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan buffer\n";
			return {};
		}
	}

	// ПРИМЕЧАНИЕ: Создание буфера не выделяет память,
	//             создаётся только **объект** буфера.
	//             Поэтому выделяем память для буфера

	{
		// ПРИМЕЧАНИЕ: Запрашиваем у буфера требования к памяти
		VkMemoryRequirements requirements;
		vkGetBufferMemoryRequirements(device, result.buffer, &requirements);

		// ПРИМЕЧАНИЕ: Запрашиваем у GPU типы памяти, которые он поддерживает
		VkPhysicalDeviceMemoryProperties properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

		// ПРИМЕЧАНИЕ: Нам нужен тип памяти, доступный как CPU, так и GPU
		// ПРИМЕЧАНИЕ: HOST - это CPU, DEVICE - это GPU; нам нужна память, видимая для CPU
		// ПРИМЕЧАНИЕ: COHERENT означает, что кэш CPU будет инвалидирован при маппинге области памяти
		const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		// ПРИМЕЧАНИЕ: Линейный поиск по типам памяти, пока
		//             один из типов не соответствует требованиям - это индекс типа памяти
		uint32_t index = UINT_MAX;
		for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
			const VkMemoryType& type = properties.memoryTypes[i];

			if ((requirements.memoryTypeBits & (1 << i)) &&
			    (type.propertyFlags & flags) == flags) {
				index = i;
				break;
			}
		}

		if (index == UINT_MAX) {
			std::cerr << "Failed to find required memory type to allocate Vulkan buffer\n";
			return {};
		}

		// ПРИМЕЧАНИЕ: Выделяем необходимый объём памяти в подходящем типе памяти
		VkMemoryAllocateInfo info{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = requirements.size,
			.memoryTypeIndex = index,
		};

		if (vkAllocateMemory(device, &info, nullptr, &result.memory) != VK_SUCCESS) {
			std::cerr << "Failed to allocate Vulkan buffer memory\n";
			return {};
		}

		// ПРИМЕЧАНИЕ: Связываем выделенную память с буфером
		if (vkBindBufferMemory(device, result.buffer, result.memory, 0) != VK_SUCCESS) {
			std::cerr << "Failed to bind Vulkan  buffer memory\n";
			return {};
		}

		// ПРИМЕЧАНИЕ: Получаем указатель на выделенную память
		void* device_data;
		vkMapMemory(device, result.memory, 0, requirements.size, 0, &device_data);

		memcpy(device_data, data, size);

		vkUnmapMemory(device, result.memory);
	}

	return result;
}

void destroyBuffer(const VulkanBuffer& buffer) {
	VkDevice& device = veekay::app.vk_device;

	vkFreeMemory(device, buffer.memory, nullptr);
	vkDestroyBuffer(device, buffer.buffer, nullptr);
}

void initialize() {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // ПРИМЕЧАНИЕ: Строим графический конвейер
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// ПРИМЕЧАНИЕ: Этап вершинного шейдера
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// ПРИМЕЧАНИЕ: Этап фрагментного шейдера
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// ПРИМЕЧАНИЕ: Сколько байт занимает одна вершина?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// ПРИМЕЧАНИЕ: Объявляем атрибуты вершин
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // ПРИМЕЧАНИЕ: Первый атрибут
				.binding = 0, // ПРИМЕЧАНИЕ: Первый вершинный буфер
				.format = VK_FORMAT_R32G32B32_SFLOAT, // ПРИМЕЧАНИЕ: 3-компонентный вектор из float
				.offset = offsetof(Vertex, position), // ПРИМЕЧАНИЕ: Смещение поля "position" в структуре Vertex
			},
			// ПРИМЕЧАНИЕ: Если нужно больше атрибутов на вершину, объявите их здесь
#if 0
			{
				.location = 1, // ПРИМЕЧАНИЕ: Второй атрибут
				.binding = 0,
				.format = VK_FORMAT_XXX,
				.offset = offset(Vertex, your_attribute),
			},
#endif
		};

		// ПРИМЕЧАНИЕ: Собираем всё вместе
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// ПРИМЕЧАНИЕ: Каждые три вершины образуют треугольник,
		//             поэтому наш вершинный буфер содержит "список треугольников"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// ПРИМЕЧАНИЕ: Объявляем порядок вершин по часовой стрелке как лицевую сторону
		//             Отбрасываем треугольники, обращённые от камеры
		//             Заполняем треугольники, а не рисуем линии вместо них
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// ПРИМЕЧАНИЕ: Используем 1 сэмпл на пиксель
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// ПРИМЕЧАНИЕ: Позволяем растеризатору рисовать на всём окне
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// ПРИМЕЧАНИЕ: Позволяем растеризатору выполнять тест глубины и перезаписывать значения глубины при успехе
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// ПРИМЕЧАНИЕ: Позволяем фрагментному шейдеру записывать все цветовые каналы
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// ПРИМЕЧАНИЕ: Позволяем растеризатору просто копировать результирующие пиксели в буфер, без смешивания
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		// ПРИМЕЧАНИЕ: Объявляем область константной памяти, видимую вершинному и фрагментному шейдерам
		VkPushConstantRange push_constants{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
			              VK_SHADER_STAGE_FRAGMENT_BIT,
			.size = sizeof(ShaderConstants),
		};

		// ПРИМЕЧАНИЕ: Объявляем внешние источники данных, только push constants в этот раз
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_constants,
		};

		// ПРИМЕЧАНИЕ: Создаём layout пайплайна
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// ПРИМЕЧАНИЕ: Создаём графический пайплайн
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	// ПРИМЕЧАНИЕ: Генерируем геометрию конуса
	const int cone_segments = 16; // Количество сегментов по окружности основания
	const float cone_radius = 1.0f;
	const float cone_height = 2.0f;
	
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	
	// Вершина центра основания (индекс 0)
	vertices.push_back({{0.0f, 0.0f, 0.0f}});
	
	// Вершины окружности основания (индексы от 1 до cone_segments)
	for (int i = 0; i < cone_segments; i++) {
		float angle = 2.0f * M_PI * i / cone_segments;
		float x = cos(angle) * cone_radius;
		float z = sin(angle) * cone_radius;
		vertices.push_back({{x, 0.0f, z}});
	}
	
	// Вершина конуса (индекс cone_segments + 1)
	vertices.push_back({{0.0f, cone_height, 0.0f}});
	
	// Генерируем индексы для треугольников основания
	for (int i = 0; i < cone_segments; i++) {
		int next = (i + 1) % cone_segments;
		// Треугольник: центр -> вершина i -> следующая вершина
		indices.push_back(0);
		indices.push_back(i + 1);
		indices.push_back(next + 1);
	}
	
	// Генерируем индексы для боковых треугольников
	int tip_index = cone_segments + 1;
	for (int i = 0; i < cone_segments; i++) {
		int next = (i + 1) % cone_segments;
		// Треугольник: вершина i -> вершина конуса -> следующая вершина
		indices.push_back(i + 1);
		indices.push_back(tip_index);
		indices.push_back(next + 1);
	}

	vertex_buffer = createBuffer(vertices.size() * sizeof(Vertex), vertices.data(),
	                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

	index_buffer = createBuffer(indices.size() * sizeof(uint32_t), indices.data(),
	                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	
	index_count = indices.size();
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	// ПРИМЕЧАНИЕ: Освобождаем ресурсы, избегаем утечек памяти
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	ImGui::Begin("Cone Controls");
	
	ImGui::SliderFloat("Rotation Speed", &rotation_speed_multiplier, 0.0f, 3.0f);
	
	ImGui::Separator();
	
	ImGui::Text("Projection Type:");
	if (ImGui::RadioButton("Perspective", use_perspective_projection)) {
		use_perspective_projection = true;
	}
	ImGui::SameLine();
	if (ImGui::RadioButton("Orthographic", !use_perspective_projection)) {
		use_perspective_projection = false;
	}
	
	ImGui::Separator();

	ImGui::Text("Animation Control:");
	if (ImGui::Button(animation_paused ? "Resume" : "Pause")) {
		animation_paused = !animation_paused;
	}
	ImGui::SameLine();
	if (ImGui::Button("Reverse Direction")) {
		animation_direction = -animation_direction;
	}
	ImGui::Text("Direction: %s", animation_direction > 0 ? "Forward" : "Reverse");
	
	ImGui::Separator();
	ImGui::Text("Cone Properties:");
	
	// Показываем свойства каждого конуса
	for (int i = 0; i < NUM_CONES; i++) {
		ImGui::PushID(i);
		char label[32];
		snprintf(label, sizeof(label), "Cone %d", i + 1);
		
		if (ImGui::TreeNode(label)) {
			ImGui::InputFloat3("Position", reinterpret_cast<float*>(&cones[i].position));
			ImGui::SliderFloat("Individual Speed", &cones[i].rotation_speed, 0.0f, 3.0f);
			ImGui::SliderFloat("Scale", &cones[i].scale, 0.1f, 2.0f);
			ImGui::Text("Current Rotation: %.2f rad", cones[i].rotation);
			ImGui::TreePop();
		}
		
		ImGui::PopID();
	}
	
	ImGui::Separator();
	ImGui::Text("Fixed Color (read-only):");
	ImGui::ColorEdit3("##FixedColor", reinterpret_cast<float*>(&fixed_color), ImGuiColorEditFlags_NoInputs);
	
	ImGui::End();

	// Обновление вращения конусов (независимо друг от друга)
	static double last_time = 0.0;
	double delta_time = time - last_time;
	last_time = time;
	
	// Применяем паузу и реверс
	if (!animation_paused) {
		for (int i = 0; i < NUM_CONES; i++) {
			cones[i].rotation += cones[i].rotation_speed * rotation_speed_multiplier * animation_direction * float(delta_time);
			cones[i].rotation = fmodf(cones[i].rotation, 2.0f * M_PI);
		}
	}
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // ПРИМЕЧАНИЕ: Начинаем запись команд рендеринга
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // ПРИМЕЧАНИЕ: Используем текущий framebuffer свопчейна и очищаем его
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	// ПРИМЕЧАНИЕ: Рисуем несколько конусов
	{
	// ПРИМЕЧАНИЕ: Используем наш графический пайплайн для конуса
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	// ПРИМЕЧАНИЕ: Привязываем вершинный и индексный буферы конуса один раз
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);
		vkCmdBindIndexBuffer(cmd, index_buffer.buffer, offset, VK_INDEX_TYPE_UINT32);

		// Вычисляем матрицу проекции один раз
		Matrix proj;
		float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
		
		if (use_perspective_projection) {
			proj = projection(camera_fov, aspect_ratio, camera_near_plane, camera_far_plane);
		} else {
			// Ортографическая проекция с фиксированными границами
			float ortho_size = 8.0f; // Размер видимой области
			float left = -ortho_size * aspect_ratio;
			float right = ortho_size * aspect_ratio;
			float bottom = -ortho_size;
			float top = ortho_size;
			proj = orthographic_projection(left, right, bottom, top, camera_near_plane, camera_far_plane);
		}

		// Рисуем каждый конус с собственной матрицей трансформации, но с фиксированным цветом
		for (int i = 0; i < NUM_CONES; i++) {
			// Вычисляем матрицу преобразования: масштаб -> вращение -> перенос
			Matrix scale_matrix = scaling(cones[i].scale);
			Matrix rot_matrix = rotation({0.0f, 1.0f, 0.0f}, cones[i].rotation);
			Matrix trans_matrix = translation(cones[i].position);
			
			Matrix transform = multiply(multiply(trans_matrix, rot_matrix), scale_matrix);

			ShaderConstants constants{
				.projection = proj,
				.transform = transform,
				.color = fixed_color,  // Фиксированный цвет для всех конусов
			};

			// Обновляем константы шейдера для этого конуса
			vkCmdPushConstants(cmd, pipeline_layout,
							   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
							   0, sizeof(ShaderConstants), &constants);

			vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
		}
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
