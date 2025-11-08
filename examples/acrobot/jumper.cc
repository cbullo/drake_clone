#include <cmath>
#include <memory>
#include <stdexcept>

#include <Eigen/Dense>

#include "drake/examples/acrobot/acrobot_geometry.h"
#include "drake/examples/acrobot/acrobot_plant.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/framework/basic_vector.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/primitives/linear_system.h"

namespace drake {
namespace examples {
namespace acrobot {

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using systems::BasicVector;
using systems::Context;
using systems::ContinuousState;
using systems::DiagramBuilder;
using systems::LeafSystem;
using systems::Simulator;

// ------------------------------------------------------------
// Simple 3D "acrobot-like" plant with:
//   q = [alpha, beta, gamma, psi, phi] (5 DOF)
//   qdot same order
// Shoulder: ball joint (alpha,beta,gamma) via Z-Y-X Euler
// Elbow: universal joint (psi,phi) via Rx(psi) then Ry(phi)
// Links: L1, L2, massless; point mass m at tip.
// Dynamics: M(q) qddot + g(q) = B u, C(q,qdot)=0.
// ------------------------------------------------------------

#include "drake/systems/controllers/linear_quadratic_regulator.h"
#include "drake/systems/primitives/linear_system.h"

using drake::examples::acrobot::AcrobotGeometry;
using drake::geometry::DrakeVisualizerd;
using drake::geometry::SceneGraph;

int DoMain();

namespace {

// ...

// Rotation helpers.
Eigen::Matrix3d Rz(double a) {
  const double c = std::cos(a), s = std::sin(a);
  Eigen::Matrix3d R;
  R << c, -s, 0, s, c, 0, 0, 0, 1;
  return R;
}

Eigen::Matrix3d Ry(double a) {
  const double c = std::cos(a), s = std::sin(a);
  Eigen::Matrix3d R;
  R << c, 0, s, 0, 1, 0, -s, 0, c;
  return R;
}

Eigen::Matrix3d Rx(double a) {
  const double c = std::cos(a), s = std::sin(a);
  Eigen::Matrix3d R;
  R << 1, 0, 0, 0, c, -s, 0, s, c;
  return R;
}

// Shoulder rotation: world <- shoulder frame
Eigen::Matrix3d ShoulderRotation(const VectorXd& q) {
  const double alpha = q(0);
  const double beta = q(1);
  const double gamma = q(2);
  return Rz(alpha) * Ry(beta) * Rx(gamma);
}

// Elbow rotation in link-1 frame.
Eigen::Matrix3d ElbowRotation(const VectorXd& q) {
  const double psi = q(3);
  const double phi = q(4);
  return Rx(psi) * Ry(phi);
}

// Tip position in world given q.
// L1, L2 will be passed as constants.
Vector3d TipPosition(const VectorXd& q, double L1, double L2) {
  const Eigen::Matrix3d Rs = ShoulderRotation(q);
  const Eigen::Matrix3d Re = ElbowRotation(q);
  const Vector3d ez(0.0, 0.0, 1.0);
  // p = Rs * (L1 * ez + Re * (L2 * ez))
  return Rs * (L1 * ez + Re * (L2 * ez));
}

// Link1 direction (shoulder -> knee) in world.
Vector3d Link1Direction(const VectorXd& q) {
  const Eigen::Matrix3d Rs = ShoulderRotation(q);
  const Vector3d ez(0.0, 0.0, 1.0);
  return (Rs * ez).normalized();
}

// Link2 direction (knee -> mass) in world.
Vector3d Link2Direction(const VectorXd& q) {
  const Eigen::Matrix3d Rs = ShoulderRotation(q);
  const Eigen::Matrix3d Re = ElbowRotation(q);
  const Vector3d ez(0.0, 0.0, 1.0);
  return (Rs * (Re * ez)).normalized();
}

// Gravity vector in world.
Vector3d GravityVector() {
  return Vector3d(0.0, 0.0, -9.81);
}

// Finite-difference Jacobian of tip position wrt q (5 DOF).
Eigen::Matrix<double, 3, 5> TipJacobian(const VectorXd& q, double L1, double L2,
                                        double eps = 1e-6) {
  Eigen::Matrix<double, 3, 5> J;
  for (int i = 0; i < 5; ++i) {
    VectorXd qp = q;
    VectorXd qm = q;
    qp(i) += eps;
    qm(i) -= eps;
    const Vector3d pp = TipPosition(qp, L1, L2);
    const Vector3d pm = TipPosition(qm, L1, L2);
    J.col(i) = (pp - pm) / (2.0 * eps);
  }
  return J;
}

// Compute instantaneous plane basis from gravity and r = tip position.
// e_r: unit vector along r
// e_p: projection of gravity into plane perpendicular to e_r
// e_n: plane normal e_r × e_p
void ComputePlaneBasis(const VectorXd& q, double L1, double L2, Vector3d* e_r,
                       Vector3d* e_p, Vector3d* e_n) {
  Vector3d r = TipPosition(q, L1, L2);
  if (r.norm() < 1e-8) {
    // Degenerate; choose arbitrary.
    *e_r = Vector3d(0, 0, 1);
  } else {
    *e_r = r.normalized();
  }

  const Vector3d g = GravityVector();
  Vector3d g_proj = g - (g.dot(*e_r)) * (*e_r);
  if (g_proj.norm() < 1e-8) {
    // r || g: choose arbitrary plane.
    Vector3d tmp(1.0, 0.0, 0.0);
    if (std::fabs(e_r->dot(tmp)) > 0.9) {
      tmp = Vector3d(0.0, 1.0, 0.0);
    }
    g_proj = tmp - tmp.dot(*e_r) * (*e_r);
  }
  *e_p = g_proj.normalized();
  *e_n = e_r->cross(*e_p);
}

void ComputePlanarAngles(const Eigen::VectorXd& q, double L1, double L2,
                         double* theta1, double* theta2) {
  using Eigen::Vector3d;

  // Plane basis from gravity and tip position.
  Vector3d e_r, e_p, e_n;
  ComputePlaneBasis(q, L1, L2, &e_r, &e_p, &e_n);

  Vector3d d1 = Link1Direction(q);  // shoulder -> knee
  Vector3d d2 = Link2Direction(q);  // knee -> mass
  Vector3d g = GravityVector();     // (0,0,-g)

  auto angle_in_plane = [&](const Vector3d& v) {
    const double vr = v.dot(e_r);
    const double vp = v.dot(e_p);
    return std::atan2(vp, vr);  // angle of v in (e_r,e_p) plane
  };

  // Absolute angles in the plane.
  const double phi1 = angle_in_plane(d1);  // link 1
  const double phi2 = angle_in_plane(d2);  // link 2
  const double phi_g = angle_in_plane(g);  // gravity direction

  // θ1: angle of link 1 relative to gravity in that plane.
  double th1 = phi1 - phi_g;

  // Wrap to (-π, π], for nice small deviations around upright.
  while (th1 <= -M_PI) th1 += 2.0 * M_PI;
  while (th1 > M_PI) th1 -= 2.0 * M_PI;

  // θ2: relative angle of link2 vs link1 in plane (same as before).
  double th2 = phi2 - phi1;
  while (th2 <= -M_PI) th2 += 2.0 * M_PI;
  while (th2 > M_PI) th2 -= 2.0 * M_PI;

  *theta1 = th1;
  *theta2 = th2;
}
// Numeric Jacobian of [theta1, theta2] wrt q (5 DOF).
Eigen::Matrix<double, 2, 5> ThetaJacobian(const VectorXd& q, double L1,
                                          double L2, double eps = 1e-6) {
  double th1_0, th2_0;
  ComputePlanarAngles(q, L1, L2, &th1_0, &th2_0);
  Eigen::Matrix<double, 2, 5> J;
  for (int i = 0; i < 5; ++i) {
    VectorXd qp = q;
    VectorXd qm = q;
    qp(i) += eps;
    qm(i) -= eps;
    double th1p, th2p, th1m, th2m;
    ComputePlanarAngles(qp, L1, L2, &th1p, &th2p);
    ComputePlanarAngles(qm, L1, L2, &th1m, &th2m);
    J(0, i) = (th1p - th1m) / (2.0 * eps);
    J(1, i) = (th2p - th2m) / (2.0 * eps);
  }
  return J;
}

// Planar angular velocities [theta1dot, theta2dot] from q, qdot.
Eigen::Vector2d ComputePlanarAngularVelocities(const VectorXd& q,
                                               const VectorXd& qdot, double L1,
                                               double L2) {
  Eigen::Matrix<double, 2, 5> Jtheta = ThetaJacobian(q, L1, L2);
  Eigen::Vector2d thetadot = Jtheta * qdot;
  return thetadot;
}

// Elbow universal joint axes in world frame.
// Axis1: x-axis of link1 frame (after shoulder rotation).
// Axis2: y-axis after first rotation Rx(psi).
void ComputeElbowAxesWorld(const VectorXd& q, Vector3d* axis1,
                           Vector3d* axis2) {
  const Eigen::Matrix3d Rs = ShoulderRotation(q);
  const double psi = q(3);
  const Eigen::Matrix3d R1 = Rx(psi);

  const Vector3d ex(1.0, 0.0, 0.0);
  const Vector3d ey(0.0, 1.0, 0.0);

  *axis1 = Rs * ex;         // first universal axis
  *axis2 = Rs * (R1 * ey);  // second universal axis
}

}  // namespace

// ------------------------------------------------------------
// 3D acrobot-like plant: 5 DOF q, 5 DOF qdot, 2 inputs at elbow.
// ------------------------------------------------------------

class Acrobot3dPlant final : public LeafSystem<double> {
 public:
  Acrobot3dPlant() {
    nq_ = 5;
    nv_ = 5;
    nx_ = nq_ + nv_;
    nu_ = 2;

    L1_ = 1.0;
    L2_ = 1.0;
    m_ = 1.0;

    // u = [u_psi, u_phi]
    this->DeclareVectorInputPort("u", BasicVector<double>(nu_));

    // continuous state x = [q; qdot]
    this->DeclareContinuousState(nx_);

    // Output: full state x, depends ONLY on state (no input feedthrough).
    this->DeclareVectorOutputPort(
        "x", BasicVector<double>(nx_), &Acrobot3dPlant::CopyStateOut,
        {this->all_state_ticket()});  // <- this kills the algebraic loop
  }

  int num_positions() const { return nq_; }
  int num_velocities() const { return nv_; }
  int num_states() const { return nx_; }
  int num_inputs() const { return nu_; }

 private:
  void CopyStateOut(const Context<double>& context,
                    BasicVector<double>* out) const {
    out->set_value(context.get_continuous_state_vector().CopyToVector());
  }

  void DoCalcTimeDerivatives(
      const Context<double>& context,
      ContinuousState<double>* derivatives) const override {
    const VectorXd x = context.get_continuous_state_vector().CopyToVector();
    const VectorXd u = this->get_input_port(0).Eval(context);

    VectorXd xdot(nx_);
    ComputeDynamics(x, u, &xdot);
    derivatives->SetFromVector(xdot);
  }

  // Dynamics: M(q) qddot + g(q) = B u, with C=0.
  void ComputeDynamics(const VectorXd& x, const VectorXd& u,
                                       VectorXd* xdot) const {
    DRAKE_DEMAND(x.size() == nq_ + nv_);
    DRAKE_DEMAND(u.size() == nu_);

    const VectorXd q = x.head(nq_);
    const VectorXd qdot = x.tail(nv_);

    // 1) Inertia matrix M(q) = m Jᵀ J + M_reg.
    Eigen::Matrix<double, 3, 5> J = TipJacobian(q, L1_, L2_);
    Eigen::Matrix<double, 5, 5> M = m_ * (J.transpose() * J);

    // Regularizing / joint inertia for all DOFs.
    // These numbers are *not* tiny on purpose – give them some real inertia.
    Eigen::Matrix<double, 5, 5> M_reg = Eigen::Matrix<double, 5, 5>::Zero();
    M_reg(0, 0) = 0.2;   // shoulder α inertia
    M_reg(1, 1) = 0.2;   // shoulder β inertia
    M_reg(2, 2) = 0.2;   // shoulder γ inertia
    M_reg(3, 3) = 0.1;  // elbow ψ inertia
    M_reg(4, 4) = 0.1;  // elbow φ inertia
    M += M_reg;

    // 2) Gravity generalized forces g(q).
    // Force on the tip mass from gravity:
    const Vector3d f_grav = m_ * GravityVector();  // (0,0,-m g)
    // Generalized gravity = Jᵀ f_grav.
    Eigen::Matrix<double, 5, 1> G = J.transpose() * f_grav;

    // 3) Joint damping D qdot.
    Eigen::Matrix<double, 5, 5> D = Eigen::Matrix<double, 5, 5>::Zero();
    D(0, 0) = 0.3;  // damp shoulder α
    D(1, 1) = 0.3;  // damp shoulder β
    D(2, 2) = 0.3;  // damp shoulder γ
    D(3, 3) = 0.2;  // damp elbow ψ
    D(4, 4) = 0.2;  // damp elbow φ

    // 4) Input mapping B u (two elbow axes).
    Eigen::Matrix<double, 5, 2> B = Eigen::Matrix<double, 5, 2>::Zero();
    B(3, 0) = 1.0;  // u_psi
    B(4, 1) = 1.0;  // u_phi

    const Eigen::Vector2d u_vec = u;

    // 5) Assemble and solve:
    //    M qddot + D qdot + G = B u   ⇒   M qddot = B u - D qdot - G
    Eigen::Matrix<double, 5, 1> rhs = B * u_vec - D * qdot - G;

    Eigen::Matrix<double, 5, 1> qddot = M.ldlt().solve(rhs);

    // 6) State derivative.
    xdot->head(nq_) = qdot;
    xdot->tail(nv_) = qddot;
  }

  int nq_{0}, nv_{0}, nx_{0}, nu_{0};
  double L1_{1.0}, L2_{1.0}, m_{1.0};
};

class Acrobot3DStateToPlanar final : public systems::LeafSystem<double> {
 public:
  // x3d_dim should match Acrobot3dPlant::num_states() (10 in our case).
  Acrobot3DStateToPlanar(double L1, double L2, int x3d_dim = 10)
      : L1_(L1), L2_(L2), nx3d_(x3d_dim) {
    this->DeclareVectorInputPort("x3d", systems::BasicVector<double>(nx3d_));
    this->DeclareVectorOutputPort("xacrobot", systems::BasicVector<double>(4),
                                  &Acrobot3DStateToPlanar::CalcOutput);
  }

 private:
  void CalcOutput(const systems::Context<double>& context,
                  systems::BasicVector<double>* out) const {
    const Eigen::VectorXd x3d = this->get_input_port(0).Eval(context);
    // Assuming q = first 5, qdot = last 5; adjust if yours differ.
    const Eigen::VectorXd q = x3d.head(5);
    const Eigen::VectorXd qdot = x3d.tail(5);

    double theta1, theta2;
    ComputePlanarAngles(q, L1_, L2_, &theta1, &theta2);

    Eigen::Vector2d thetadot =
        ComputePlanarAngularVelocities(q, qdot, L1_, L2_);
    const double theta1dot = thetadot(0);
    const double theta2dot = thetadot(1);

    Eigen::Vector4d x_planar;
    x_planar << theta1, theta2, theta1dot, theta2dot;

    out->SetFromVector(x_planar);
  }

  double L1_{1.0}, L2_{1.0};
  int nx3d_{10};
};

// ------------------------------------------------------------
// Moving-plane acrobot controller:
//   - Uses planar AcrobotPlant LQR
//   - Maps 3D state -> planar (theta1,theta2,theta1dot,theta2dot)
//   - Computes scalar torque tau_plane
//   - Maps tau_plane -> 3D elbow torques (u_psi,u_phi)
// ------------------------------------------------------------

class MovingPlaneAcrobotController final : public LeafSystem<double> {
 public:
  MovingPlaneAcrobotController(double L1, double L2) : L1_(L1), L2_(L2) {
    nx3d_ = 10;  // 5 q + 5 qdot
    nu3d_ = 2;

    this->DeclareVectorInputPort("x3d", BasicVector<double>(nx3d_));
    this->DeclareVectorOutputPort("u", BasicVector<double>(nu3d_),
                                  &MovingPlaneAcrobotController::CalcOutput);

    BuildPlanarLqr();
  }

 private:
  void BuildPlanarLqr() {
    using systems::Context;
    using systems::Linearize;
    using systems::LinearSystem;
    using systems::controllers::LinearQuadraticRegulator;
    using systems::controllers::LinearQuadraticRegulatorResult;

    // Planar AcrobotPlant from Drake.
    AcrobotPlant<double> acrobot;
    auto context = acrobot.CreateDefaultContext();

    // Upright equilibrium: link 1 up, link 2 aligned with link 1.
    // For Drake's AcrobotPlant, that is typically [theta1, theta2] = [π, 0].
    Eigen::Vector4d x0;
    x0 << M_PI, 0.0, 0.0, 0.0;  // [θ1, θ2, θ̇1, θ̇2]
    context->SetContinuousState(x0);

    // Zero torque at equilibrium.
    acrobot.get_input_port().FixValue(context.get(), 0.0);

    // Linearize around this equilibrium.
    std::unique_ptr<LinearSystem<double>> lin_sys =
        Linearize(acrobot, *context);

    const Eigen::MatrixXd& A = lin_sys->A();
    const Eigen::MatrixXd& B = lin_sys->B();

    // State cost for [θ1, θ2, θ̇1, θ̇2] (tune as you like).
    Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
    Q(0, 0) = 10.0;  // θ1 error (upright)
    Q(1, 1) = 5.0;  // θ2 error
    Q(2, 2) = 1.0;   // θ̇1
    Q(3, 3) = 0.5;   // θ̇2

    Eigen::Matrix<double, 1, 1> R;
    R(0, 0) = 0.5;

    Eigen::MatrixXd N;  // zero
    Eigen::MatrixXd F;  // zero

    LinearQuadraticRegulatorResult result =
        LinearQuadraticRegulator(A, B, Q, R, N, F);

    // Feedback is on deviation x̄ = x - x0: ū = -K x̄, u = u0 + ū, u0=0.
    K_planar_ = result.K;  // 1x4
    x0_planar_ = x0;       // store upright eq
    u0_planar_.setZero();  // scalar 0
  }

  void CalcOutput(const Context<double>& context,
                  BasicVector<double>* out) const {
    const VectorXd x3d = this->get_input_port(0).Eval(context);

    const VectorXd q = x3d.head(5);
    const VectorXd qdot = x3d.tail(5);

    // 1) Plane basis from gravity and tip vector.
    Vector3d e_r, e_p, e_n;
    ComputePlaneBasis(q, L1_, L2_, &e_r, &e_p, &e_n);

    // 2) Planar acrobot state.
    double theta1, theta2;
    ComputePlanarAngles(q, L1_, L2_, &theta1, &theta2);

    Eigen::Vector2d thetadot =
        ComputePlanarAngularVelocities(q, qdot, L1_, L2_);
    const double theta1dot = thetadot(0);
    const double theta2dot = thetadot(1);

    Eigen::Vector4d x_planar;
    x_planar << theta1, theta2, theta1dot, theta2dot;

    // Control about upright: u = u0 - K (x - x0).
    Eigen::Matrix<double, 1, 1> u_planar =
        u0_planar_ - K_planar_ * (x_planar - x0_planar_);
    const double tau_plane = u_planar(0);

    // 4) Map tau_plane about e_n to elbow axes.
    Vector3d axis1, axis2;
    ComputeElbowAxesWorld(q, &axis1, &axis2);

    Eigen::Matrix<double, 3, 2> A;
    A.col(0) = axis1;
    A.col(1) = axis2;

    Eigen::Vector2d u_elbow = A.colPivHouseholderQr().solve(tau_plane * e_n);

    VectorXd u3d(2);
    u3d << u_elbow(0), u_elbow(1);
    out->SetFromVector(u3d);
  }
  int nx3d_{0}, nu3d_{0};
  double L1_{1.0}, L2_{1.0};

  // Planar LQR gain (1x4).
  Eigen::RowVector4d K_planar_;
  Eigen::Vector4d x0_planar_;
  Eigen::Matrix<double, 1, 1> u0_planar_;
};

// ------------------------------------------------------------
// Example main: 3D plant + moving-plane acrobot controller.
// ------------------------------------------------------------

int DoMain() {
  systems::DiagramBuilder<double> builder;

  const double L1 = 1.0;
  const double L2 = 1.0;
  // const double m  = 1.0;
  // const double g  = 9.81;

  auto* plant = builder.AddSystem<Acrobot3dPlant>();
  plant->set_name("acrobot3d");

  auto* controller = builder.AddSystem<MovingPlaneAcrobotController>(L1, L2);
  controller->set_name("controller");

  // Plant ↔ controller connections (you already had this):
  builder.Connect(plant->get_output_port(0),       // x3d
                  controller->get_input_port(0));  // x3d
  builder.Connect(controller->get_output_port(0),  // u
                  plant->get_input_port(0));       // u

  // --- NEW: SceneGraph for visualization ---
  auto* scene_graph = builder.AddSystem<SceneGraph>();
  scene_graph->set_name("scene_graph");

  // --- NEW: adapter from 3D state to planar acrobot state ---
  auto* state_adapter =
      builder.AddSystem<Acrobot3DStateToPlanar>(L1, L2, plant->num_states());
  state_adapter->set_name("acrobot_3d_to_planar");

  builder.Connect(plant->get_output_port(0),          // x3d
                  state_adapter->get_input_port(0));  // x3d

  // --- NEW: reuse AcrobotGeometry from the example ---
  AcrobotGeometry::AddToBuilder(
      &builder,
      state_adapter->get_output_port(0),  // 4-dim planar acrobot state
      scene_graph);

  // --- NEW: connect SceneGraph to drake_visualizer ---
  DrakeVisualizerd::AddToBuilder(&builder, *scene_graph);

  // Build and simulate (same as before).
  auto diagram = builder.Build();
  systems::Simulator<double> simulator(*diagram);

  auto& context = simulator.get_mutable_context();
  auto& plant_context = diagram->GetMutableSubsystemContext(*plant, &context);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(plant->num_states());
  // e.g., small elbow perturbation:
  x0(0) = M_PI;
  x0(3) = 0.25;
  plant_context.SetContinuousState(x0);

  simulator.set_target_realtime_rate(1.0);
  simulator.Initialize();
  simulator.AdvanceTo(10.0);

  return 0;
}
}  // namespace acrobot
}  // namespace examples
}  // namespace drake

int main() {
  return drake::examples::acrobot::DoMain();
}