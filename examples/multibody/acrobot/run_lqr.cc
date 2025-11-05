#include <memory>
#include <string>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/benchmarks/acrobot/make_acrobot_plant.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/universal_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/affine_system.h"
#include "drake/visualization/visualization_config_functions.h"

namespace drake {

using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;
using geometry::SceneGraph;
using multibody::AddMultibodyPlantSceneGraph;
using multibody::JointActuator;
using multibody::MultibodyPlant;
using multibody::Parser;
using multibody::RevoluteJoint;
using multibody::UniversalJoint;
using multibody::benchmarks::acrobot::AcrobotParameters;
using multibody::benchmarks::acrobot::MakeAcrobotPlant;
using systems::Context;
using systems::InputPort;

namespace examples {
namespace multibody {
namespace acrobot {
namespace {

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");

DEFINE_double(simulation_time, 3.0,
              "Desired duration of the simulation in seconds.");

DEFINE_bool(time_stepping, true,
            "If 'true', the plant is modeled as a discrete system with "
            "periodic updates. "
            "If 'false', the plant is modeled as a continuous system.");

// // This helper method makes an LQR controller to balance an acrobot model
// // specified in the SDF file `file_name`.
std::unique_ptr<systems::AffineSystem<double>> MakeBalancingLQRController(
    const std::string& acrobot_url) {
  // LinearQuadraticRegulator() below requires the controller's model of the
  // plant to only have a single input port corresponding to the actuation.
  // Therefore we create a new model that meets this requirement. (a model
  // created along with a SceneGraph for simulation would also have input
  // ports
  // to interact with that SceneGraph).
  MultibodyPlant<double> acrobot(0.0);
  Parser parser(&acrobot);
  parser.AddModelsFromUrl(acrobot_url);
  // We are done defining the model.
  acrobot.Finalize();

//   UniversalJoint<double>& shoulder =
//       acrobot.GetMutableJointByName<UniversalJoint>("ShoulderJoint");
  RevoluteJoint<double>& elbow_a =
      acrobot.GetMutableJointByName<RevoluteJoint>("ElbowJointA");
  RevoluteJoint<double>& elbow_b =
      acrobot.GetMutableJointByName<RevoluteJoint>("ElbowJointB");

  std::unique_ptr<Context<double>> context = acrobot.CreateDefaultContext();

  // Set nominal actuation torque to zero.
  const InputPort<double>& actuation_port = acrobot.get_actuation_input_port();
  actuation_port.FixValue(context.get(), Vector2d{0.0, 0.0});
  acrobot.get_applied_generalized_force_input_port().FixValue(
      context.get(), Vector4d::Constant(0.0));

//   shoulder.set_angles(context.get(), {0.0, 0.0});
//   shoulder.set_angular_rates(context.get(), {0.0, 0.0});
  elbow_a.set_angle(context.get(), 0.0);
  elbow_a.set_angular_rate(context.get(), 0.0);
  elbow_b.set_angle(context.get(), 0.0);
  elbow_b.set_angular_rate(context.get(), 0.0);

  //   shoulder.set_default_angles({M_PI, 0.0});
  //   elbow_a.set_default_angle(0.0);
  //   elbow_b.set_default_angle(0.0);

  // Setup LQR Cost matrices (penalize position error 10x more than velocity
  // to roughly address difference in units, using sqrt(g/l) as the time
  // constant.
  Eigen::Matrix<double, 8, 8> Q = Eigen::Matrix<double, 8, 8>::Identity();
  Q(0, 0) = 10.0;
  Q(1, 1) = 10.0;
  Q(2, 2) = 10.0;
  Q(3, 3) = 10.0;

  Eigen::Matrix2d R;
  R(0, 0) = 1.0;
  R(1, 1) = 1.0;
  R(0, 1) = 0.05;
  R(1, 0) = 0.05;

  return systems::controllers::LinearQuadraticRegulator(
      acrobot, *context, Q, R,
      Eigen::Matrix<double, 0, 0>::Zero() /* No cross state/control costs */,
      actuation_port.get_index());
}

int do_main() {
  systems::DiagramBuilder<double> builder;

  const double time_step = FLAGS_time_stepping ? 1.0e-3 : 0.0;
  auto [acrobot, scene_graph] =
      AddMultibodyPlantSceneGraph(&builder, time_step);

  // Make and add the acrobot model.
  const std::string acrobot_url =
      "package://drake/multibody/benchmarks/acrobot/acrobot.sdf";
  Parser parser(&builder);
  parser.AddModelsFromUrl(acrobot_url);

  // We are done defining the model.
  acrobot.Finalize();

  DRAKE_DEMAND(acrobot.num_actuators() == 2);
  DRAKE_DEMAND(acrobot.num_actuated_dofs() == 2);

  //    UniversalJoint<double>& shoulder =
  //        acrobot.GetMutableJointByName<UniversalJoint>("ShoulderJoint");
  RevoluteJoint<double>& elbow_a =
      acrobot.GetMutableJointByName<RevoluteJoint>("ElbowJointA");
  RevoluteJoint<double>& elbow_b =
      acrobot.GetMutableJointByName<RevoluteJoint>("ElbowJointB");

  // Drake's parser will default the name of the actuator to match the name of
  // the joint it actuates.
  const JointActuator<double>& actuator_a =
      acrobot.GetJointActuatorByName("ElbowJointA");
  DRAKE_DEMAND(actuator_a.joint().name() == "ElbowJointA");

  const JointActuator<double>& actuator_b =
      acrobot.GetJointActuatorByName("ElbowJointB");
  DRAKE_DEMAND(actuator_b.joint().name() == "ElbowJointB");

  // For this example the controller's model of the plant exactly matches the
  // plant to be controlled (in reality there would always be a mismatch).
  auto controller = builder.AddSystem(MakeBalancingLQRController(acrobot_url));
  controller->set_name("controller");
  builder.Connect(acrobot.get_state_output_port(),
                  controller->get_input_port());
  builder.Connect(controller->get_output_port(),
                  acrobot.get_actuation_input_port());

  visualization::AddDefaultVisualization(&builder);
  auto diagram = builder.Build();

  systems::Simulator<double> simulator(*diagram);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);

  RandomGenerator generator;

  // Setup distribution for random initial conditions.
  std::normal_distribution<symbolic::Expression> gaussian(0.0, 0.006);
  elbow_a.set_random_angle_distribution({gaussian(generator)});
  elbow_b.set_random_angle_distribution({gaussian(generator)});

  for (int i = 0; i < 5; i++) {
    simulator.get_mutable_context().SetTime(0.0);
    simulator.get_system().SetRandomContext(&simulator.get_mutable_context(),
                                            &generator);

    simulator.Initialize();
    simulator.AdvanceTo(FLAGS_simulation_time);
  }

  return 0;
}

}  // namespace
}  // namespace acrobot
}  // namespace multibody
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "A simple acrobot demo using Drake's MultibodyPlant with "
      "LQR stabilization. "
      "Launch meldis before running this example.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::examples::multibody::acrobot::do_main();
}
