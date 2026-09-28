#!/usr/bin/env bash
# One-time IAM setup for the A1 hunter. Run in OCI Cloud Shell by the account
# owner: it changes who may do what in the tenancy, so it is never run by code.
#
# Grants exactly one instance (the hub VM, via instance principal) the right to
# launch instances. It cannot terminate them, and no API key is stored on the VM.
set -euo pipefail

: "${HUB_INSTANCE_ID:?set HUB_INSTANCE_ID to the hub VM OCID}"
TENANCY="${OCI_TENANCY:?run this in OCI Cloud Shell}"
NAME=redmond-a1-hunter

oci iam dynamic-group create --name "$NAME" \
  --description "Redmond hub VM: may launch the A1 instance" \
  --matching-rule "instance.id = '${HUB_INSTANCE_ID}'"

oci iam policy create --compartment-id "$TENANCY" --name "$NAME" \
  --description "A1 hunter: launch instances, never terminate" \
  --statements "[
    \"Allow dynamic-group ${NAME} to manage instance-family in tenancy where request.operation != 'TerminateInstance'\",
    \"Allow dynamic-group ${NAME} to use volume-family in tenancy\",
    \"Allow dynamic-group ${NAME} to use virtual-network-family in tenancy\"
  ]"

echo "Done. Permissions can take a minute to apply."
