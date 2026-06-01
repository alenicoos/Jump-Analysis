import Foundation

extension Date {
    var airPoseMediumString: String {
        formatted(date: .abbreviated, time: .shortened)
    }
}

extension Double {
    var airPoseOneDecimalString: String {
        formatted(.number.precision(.fractionLength(1)))
    }

    var airPoseProtocolValueString: String {
        let magnitude = abs(self)
        if magnitude >= 10 {
            return formatted(.number.precision(.fractionLength(1)))
        }
        if magnitude >= 1 {
            return formatted(.number.precision(.fractionLength(2)))
        }
        return formatted(.number.precision(.fractionLength(3)))
    }

    var airPoseCentimeterString: String {
        "\(airPoseOneDecimalString) cm"
    }

    var airPoseMillisecondsString: String {
        "\(airPoseOneDecimalString) ms"
    }

    var airPoseDegreeString: String {
        "\(airPoseOneDecimalString)°"
    }

    var airPoseScoreString: String {
        airPoseOneDecimalString
    }

    var airPoseRatioString: String {
        airPoseOneDecimalString
    }
}

extension Bool {
    var airPosePassFailString: String {
        self ? "Pass" : "Fail"
    }
}
